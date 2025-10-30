from typing import Tuple

import numpy as np

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")

    res[:, shape[1] // 2:] = -1
    res[:, :shape[1]//2] =1

    # Generar valores de 0 a 1 para la parte izquierda de la imagen
    # Un obstáculo aquí (valores altos) no afecta mucho la velocidad.
    #left_gradient = np.linspace(0, 1, shape[1] // 2)
    #res[:, : shape[1] // 2] = left_gradient
    
    # Generar valores de 0 a -1 para la parte derecha de la imagen
    # Un obstáculo aquí (valores altos) reduce significativamente la velocidad.
    #right_gradient = np.linspace(0, -1, shape[1] // 2)
    #res[:, shape[1] // 2:] = right_gradient
    
    # --- Líneas comentadas del código original ---
    #res[shape[0] // 2:, :shape[1] // 2] = np.linspace(1, 0.5, shape[1] // 2)
    #res[shape[0] // 2:, shape[1] // 2:] = np.linspace(-1, -0.5, shape[1] // 2)
    
    # Nota: La siguiente línea en el original no modifica 'res' porque el resultado no se asigna.
    # Se deja para ser fiel a la imagen.
    #res/np.sum(res)
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")

    res[:, :shape[1] //2] = -1
    res[:, shape[1]//2:] = 1

    # Generar valores de -1 a 0 para la parte izquierda de la imagen
    # Un obstáculo aquí (valores altos) reduce significativamente la velocidad.
    #left_gradient = np.linspace(-1, 0, shape[1] // 2)
    #res[:, :shape[1] // 2] = left_gradient
    
    # Generar valores de 1 a 0 para la parte derecha de la imagen
    # Un obstáculo aquí (valores altos) no afecta mucho la velocidad.
    #right_gradient = np.linspace(1, 0, shape[1] // 2)
    #res[:, shape[1] // 2:] = right_gradient
    
    # --- Líneas comentadas del código original ---
    #res[shape[0] // 2:, :shape[1] // 2] = np.linspace(0.5, 1, shape[1] // 2)
    #res[shape[0] // 2:, shape[1] // 2:] = np.linspace(-0.5, -1, shape[1] // 2)
    # res[200:400, 200:400] = 0

    # Nota: La siguiente línea en el original no modifica 'res' porque el resultado no se asigna.
    # Se deja para ser fiel a la imagen.
    #res/np.sum(res)   
    return res
