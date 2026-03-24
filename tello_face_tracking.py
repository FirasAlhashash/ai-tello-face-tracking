from djitellopy import Tello
import cv2
import numpy as np
import time
import os
import sys


# EINSTELLUNGEN
FRAME_W = 640
FRAME_H = 480

# Gesichtsfläche (in Pixel)
# kleiner -> Drohne fliegt vor, größer -> Drohne fliegt zurück
FB_RANGE = [8000, 15000]

# PID für Drehung (yaw)
PID_YAW = [0.35, 0.25, 0.0]

# PID für Höhe 
PID_UD = [0.30, 0.20, 0.0]

MAX_YAW_SPEED = 40
MAX_UD_SPEED = 25
FB_SPEED = 18

USE_VERTICAL_TRACKING = False  # True, wenn Höhenregelung gewünscht


# CASCADE LADEN
def load_face_cascade():
    """
    Lädt die Haarcascade-Datei:
    1. Versuch: lokale Datei im Skript-Ordner
    2. Versuch: OpenCV-Standardpfad (nach Installation von opencv-python)
    """
    # Mögliche Pfade
    local_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    opencv_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    # 1. lokale Datei
    if os.path.exists(local_path):
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            print(f"[INFO] Nutze lokale Cascade: {local_path}")
            return cascade
        else:
            print("[WARN] Lokale Cascade existiert, konnte aber nicht geladen werden.")

    # 2. OpenCV-Cascade
    cascade = cv2.CascadeClassifier(opencv_path)
    if cascade.empty():
        print("[ERROR] Konnte keine Gesichtserkennungs-Cascade laden.")
        print("Stelle sicher, dass opencv-python richtig installiert ist.")
        sys.exit(1)

    print(f"[INFO] Nutze OpenCV-Cascade: {opencv_path}")
    return cascade


face_cascade = load_face_cascade()


# GESICHTSERKENNUNG
def find_face(img):
    """
    Findet das größte Gesicht im Bild.
    Rückgabe:
        img: Bild mit Markierung
        info: [[cx, cy], Fläche, (x, y, w, h)]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)

    face_centers = []
    face_areas = []
    face_boxes = []

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        area = w * h

        face_centers.append([cx, cy])
        face_areas.append(area)
        face_boxes.append((x, y, w, h))

    if face_areas:
        i = face_areas.index(max(face_areas))
        x, y, w, h = face_boxes[i]
        cx, cy = face_centers[i]
        area = face_areas[i]

        # Markierungen zeichnen
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return img, [[cx, cy], area, (x, y, w, h)]
    else:
        return img, [[0, 0], 0, (0, 0, 0, 0)]



# REGELUNG
def track_face(tello, info, frame_w, frame_h, prev_yaw_error, prev_ud_error):
    """
    Berechnet die Steuerbefehle basierend auf der Gesichtsposition.
    """
    cx, cy = info[0]
    area = info[1]

    # Yaw (Drehung)
    yaw_error = cx - frame_w // 2
    yaw_speed = PID_YAW[0] * yaw_error + PID_YAW[1] * (yaw_error - prev_yaw_error)
    yaw_speed = int(np.clip(yaw_speed, -MAX_YAW_SPEED, MAX_YAW_SPEED))

    # Vor/Zurück (basierend auf Fläche)
    fb = 0
    if FB_RANGE[0] < area < FB_RANGE[1]:
        fb = 0
    elif area > FB_RANGE[1]:
        fb = -FB_SPEED
    elif 0 < area < FB_RANGE[0]:
        fb = FB_SPEED

    # Höhe 
    ud = 0
    ud_error = cy - frame_h // 2
    if USE_VERTICAL_TRACKING:
        ud_speed = PID_UD[0] * ud_error + PID_UD[1] * (ud_error - prev_ud_error)
        ud_speed = int(np.clip(ud_speed, -MAX_UD_SPEED, MAX_UD_SPEED))
        ud = -ud_speed  # wenn Gesicht zu tief, Drohne runter
    else:
        ud_error = 0

    # Wenn kein Gesicht erkannt
    if cx == 0:
        yaw_speed = 0
        fb = 0
        ud = 0
        yaw_error = 0
        ud_error = 0

    # Befehl senden
    tello.send_rc_control(0, fb, ud, yaw_speed)

    return yaw_error, ud_error, yaw_speed, fb, ud



# HAUPTFUNKTION
def main():
    tello = Tello()

    is_flying = False
    stream_started = False

    try:
        print("[INFO] Verbinde mit Tello...")
        tello.connect()
        battery = tello.get_battery()
        print(f"[INFO] Akku: {battery}%")

        if battery < 15:
            print("[WARN] Akku sehr niedrig. Bitte laden.")
            return

        # Stream starten
        print("[INFO] Starte Videostream...")
        tello.streamon()
        stream_started = True
        time.sleep(2)

        frame_read = tello.get_frame_read()

        # Kurzer Test, ob Frames ankommen
        start_wait = time.time()
        frame_ok = False
        while time.time() - start_wait < 8:
            frame = frame_read.frame
            if frame is not None and frame.size != 0:
                frame_ok = True
                break
            time.sleep(0.1)

        if not frame_ok:
            print("[ERROR] Kein Videobild empfangen.")
            print("Prüfe WLAN-Verbindung zur Tello und schließe andere Apps.")
            return

        print("\n[INFO] Steuerung:")
        print("   t  = takeoff")
        print("   l  = land")
        print("   q  = quit\n")

        prev_yaw_error = 0
        prev_ud_error = 0

        while True:
            frame = frame_read.frame

            if frame is None or frame.size == 0:
                # Leeres Bild anzeigen
                blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                cv2.putText(blank, "Kein Kamerabild", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Tello Face Tracking", blank)
            else:
                img = cv2.resize(frame, (FRAME_W, FRAME_H))
                img, info = find_face(img)

                # Bildmitte einzeichnen
                cv2.circle(img, (FRAME_W // 2, FRAME_H // 2), 5, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 0, 0), 1)
                cv2.line(img, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (255, 0, 0), 1)

                yaw_speed = fb = ud = 0

                if is_flying:
                    prev_yaw_error, prev_ud_error, yaw_speed, fb, ud = track_face(
                        tello, info, FRAME_W, FRAME_H, prev_yaw_error, prev_ud_error
                    )

                # Informationen einblenden
                face_x, face_y = info[0]
                face_area = info[1]
                cv2.putText(img, f"Battery: {tello.get_battery()}%", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Face center: ({face_x}, {face_y})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Face area: {face_area}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Yaw: {yaw_speed} | FB: {fb} | UD: {ud}", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Flying: {is_flying}", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Tello Face Tracking", img)

            # Tastaturabfrage
            key = cv2.waitKey(1) & 0xFF

            if key == ord('t') and not is_flying:
                print("[INFO] Takeoff...")
                tello.takeoff()
                time.sleep(2)
                is_flying = True

            elif key == ord('l') and is_flying:
                print("[INFO] Landing...")
                tello.land()
                is_flying = False

            elif key == ord('q'):
                print("[INFO] Beende Programm...")
                break

    except Exception as e:
        print(f"[FEHLER] {e}")

    finally:
        # Aufräumen
        if is_flying:
            print("[INFO] Sicherheitslandung...")
            try:
                tello.land()
                time.sleep(2)
            except:
                pass

        if stream_started:
            tello.streamoff()

        tello.end()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()