# Lector alfabeto lenguaje signos

Programa de lectura de alfabeto lenguaje de signos.

Utilizando la librería de cv2, opencv y keras. El modelo de aprendizaje utilizado para la creación de la lectura de manos ha sido a través de la página [TeacheableMachine](https://teachablemachine.withgoogle.com/train/image).

# Instalación:
    pip install opencv-python
    pip install mediapipe
    pip install cvzone
    pip install keras
    pip install tensorflow

# Listado de ficheros

- **Carpeta Data:** Esta carpeta contiene todas las letras del abecedario, con un total de 100 imágenes cada una siendo 50 de la mano izquierda y 50 de la mano derecha.

- **Carpeta Models:** Esta carpeta dispone de todos los posibles modelos ya entrenados y las etiquetas (labels) asociados.

- **hands.py:** Este script es una versión base para comprobar la funcionalidad de la cámara y la detección de manos.

- **handsSaveLetter.py:** Este script sirve para crear nuevas imágenes a las letras asociadas, para ello hay que modificar unos parámetros del script que son la línea 53 para la ruta donde se va a guardar la imagen.
    > Para guardar nuevas imágenes hay que ejecutar el script y al posicionar la mano en la forma adecuada pulsar la letra "s" en el teclado para guardar la imagen. (Hay una confirmación de guardado en la terminal cada vez que se guarda una foto)

- **Test.py:** Este script es el principal, solo tenemos que ejeutar el script y comenzará a leer el lenguaje de signos.
