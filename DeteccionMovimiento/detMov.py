import cv2

# Inicializar la cámara (0 = cámara predeterminada)
cap = cv2.VideoCapture(0)

# Leer el primer frame y convertirlo a escala de grises
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    # Leer el siguiente frame
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calcular la diferencia entre los dos frames
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encontrar contornos (áreas de movimiento)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Ignorar movimientos pequeños
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con detección
    cv2.imshow("Detección de Movimiento", frame2)

    # Actualizar el frame anterior
    gray1 = gray2.copy()

    # Salir con la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()