import cv2                    # Librería para procesamiento de imágenes y video
import time                   # Para medir tiempos y controlar duraciones
import os                     # Para manejar archivos y carpetas
from datetime import datetime # Para obtener la fecha y hora actual

# Parámetros configurables
THRESHOLD_VALUE = 12          # Umbral de diferencia entre píxeles para detectar movimiento
MIN_AREA = 100                # Área mínima de un contorno para considerarse movimiento
VIDEO_DURATION = 3            # Duración del video a grabar al detectar movimiento (en segundos)
FRAME_RATE = 30               # Fotogramas por segundo del video
VIDEO_WIDTH = 640             # Ancho de los frames de video
VIDEO_HEIGHT = 480            # Altura de los frames de video

last_video_time = 0           # Marca de tiempo del último video grabado

# Inicializar cámara (0 = cámara predeterminada del sistema)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)     # Establecer ancho del video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)   # Establecer alto del video

# Leer el primer frame de la cámara y convertirlo a escala de grises con desenfoque
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)        # Convertir a escala de grises
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)            # Aplicar desenfoque gaussiano para reducir ruido

# Bucle principal
while True:
    ret, frame2 = cap.read()                            # Capturar el siguiente frame
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)    # Convertir a escala de grises
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)        # Aplicar desenfoque

    # Calcular la diferencia absoluta entre los frames actuales y anteriores
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]  # Umbral binario
    thresh = cv2.dilate(thresh, None, iterations=2)      # Dilatar para unir regiones cercanas

    # Detectar contornos en la imagen binaria (posibles áreas con movimiento)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False          # Bandera para saber si se detectó movimiento
    rectangles = []                  # Lista para guardar rectángulos donde se detectó movimiento

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:  # Ignorar áreas pequeñas (ruido)
            continue
        (x, y, w, h) = cv2.boundingRect(contour)  # Obtener coordenadas del rectángulo
        rectangles.append((x, y, w, h))           # Guardar rectángulo para imagen futura
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar rectángulo verde
        motion_detected = True                    # Se ha detectado movimiento

    if motion_detected:
        current_time = time.time()
        # Solo grabar si han pasado al menos 3 segundos desde el último video
        if current_time - last_video_time >= VIDEO_DURATION:
            # Crear carpeta con la fecha actual (ej. "2025-07-05/")
            date_folder = datetime.now().strftime("%Y-%m-%d")
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)

            # Registrar el evento en el archivo log.txt
            with open("log.txt", "a") as log_file:
                log_file.write(f"Movimiento detectado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # === GUARDAR IMAGEN CON RECTÁNGULOS ===
            snapshot = frame2.copy()  # Copiar el frame actual
            for (x, y, w, h) in rectangles:
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2) # Dibujar rectángulos rojos
            img_name = datetime.now().strftime("captura_%H%M%S.jpg")            # Nombre del archivo
            img_path = os.path.join(date_folder, img_name)                      # Ruta completa
            cv2.imwrite(img_path, snapshot)                                     # Guardar la imagen
            print(f"[INFO] Imagen guardada: {img_path}")

            # === INICIAR GRABACIÓN DE VIDEO ===
            time_str = datetime.now().strftime("video_%H%M%S.avi")            # Nombre del video
            video_path = os.path.join(date_folder, time_str)                  # Ruta completa

            fourcc = cv2.VideoWriter_fourcc(*'XVID')                          # Códec de compresión
            out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (VIDEO_WIDTH, VIDEO_HEIGHT))  # Inicializar grabador

            print(f"[INFO] Grabando video: {video_path}")
            start_time = time.time()

            # Grabar durante VIDEO_DURATION segundos
            while time.time() - start_time < VIDEO_DURATION:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)                        # Escribir frame en el archivo
                cv2.imshow("Detección de Movimiento", frame)  # Mostrar el frame grabado
                if cv2.waitKey(1) & 0xFF == ord('q'):   # Salir si se presiona 'q'
                    break

            out.release()                              # Finalizar grabación
            last_video_time = time.time()              # Actualizar última grabación
            print("[INFO] Grabación finalizada.")

    # Mostrar en pantalla el frame procesado con rectángulos verdes
    cv2.imshow("Detección de Movimiento", frame2)

    # Actualizar el frame anterior para la próxima comparación
    gray1 = gray2.copy()

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas al salir del bucle
cap.release()
cv2.destroyAllWindows()