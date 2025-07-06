import cv2                     # Librería para procesamiento de imágenes y video
import time                    # Para trabajar con tiempos (pausas, marcas de tiempo)
import os                      # Para crear carpetas y trabajar con archivos
from datetime import datetime  # Para obtener la fecha y hora actuales

# Parámetros configurables
THRESHOLD_VALUE = 12           # Sensibilidad del umbral para detectar diferencias entre frames
MIN_AREA = 100                 # Área mínima (en píxeles) para considerar que un contorno representa movimiento
VIDEO_DURATION = 3             # Duración del video grabado cuando se detecta movimiento (en segundos)
FRAME_RATE = 30                # Cuántos cuadros por segundo tendrá el video
VIDEO_WIDTH = 640              # Ancho del video (en píxeles)
VIDEO_HEIGHT = 480             # Alto del video (en píxeles)

last_video_time = 0            # Marca de tiempo del último video grabado (usado para evitar duplicados)

# Inicializar la cámara (0 = cámara predeterminada)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)     # Establece el ancho del video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)   # Establece el alto del video

# Leer el primer frame y convertirlo a escala de grises con desenfoque
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)           # Convertir a escala de grises
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)               # Aplicar desenfoque para reducir el ruido

# Bucle principal
while True:
    # Capturar siguiente frame y procesarlo igual que el anterior
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calcular la diferencia entre los dos frames
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]  # Umbral binario
    thresh = cv2.dilate(thresh, None, iterations=2)                           # Expandir áreas blancas para cubrir más región

    # Buscar contornos (bordes de áreas donde hubo diferencia significativa)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False  # Indicador de si se ha detectado movimiento en este ciclo

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:  # Ignorar movimientos muy pequeños
            continue
        # Si se encuentra un contorno suficientemente grande:
        (x, y, w, h) = cv2.boundingRect(contour)  # Obtener coordenadas del rectángulo alrededor del movimiento
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar rectángulo verde sobre el movimiento
        motion_detected = True  # Se ha detectado movimiento

    if motion_detected:
        current_time = time.time()
        # Solo grabar si han pasado al menos VIDEO_DURATION segundos desde el último video
        if current_time - last_video_time >= VIDEO_DURATION:
            # Registrar en log.txt la fecha y hora del evento
            with open("log.txt", "a") as log_file:
                log_file.write(f"Movimiento detectado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Crear carpeta con la fecha actual si no existe (ej: '2025-07-05/')
            date_folder = datetime.now().strftime("%Y-%m-%d")
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)

            # Crear nombre del archivo de video dentro de la carpeta de fecha
            time_str = datetime.now().strftime("video_%H%M%S.avi")
            video_path = os.path.join(date_folder, time_str)

            # Inicializar el objeto para grabar video con el códec XVID
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (VIDEO_WIDTH, VIDEO_HEIGHT))

            print(f"[INFO] Grabando video: {video_path}")

            # Grabar durante VIDEO_DURATION segundos
            start_time = time.time()
            while time.time() - start_time < VIDEO_DURATION:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)  # Escribir cada frame en el archivo de video
                cv2.imshow("Detección de Movimiento", frame)  # Mostrar lo que se está grabando
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()                 # Finalizar y guardar el archivo de video
            last_video_time = time.time() # Actualizar la última vez que se grabó
            print("[INFO] Grabación finalizada.")

    # Mostrar frame actual con detección (incluso si no se está grabando)
    cv2.imshow("Detección de Movimiento", frame2)

    # Actualizar el frame anterior con el actual para la próxima comparación
    gray1 = gray2.copy()

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas al finalizar
cap.release()
cv2.destroyAllWindows()