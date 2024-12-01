# MachineProyect
## COMO UTILIZAR 
Para poder utilizar esta repo sin necesidad de re-entrenar el modelo se tiene que ir al archivo main
En el archivo main encontraras el codigo entero con el uso del modelo ya entrenado, lo que se hizo fue cargar en una variable las predicciones ya guardadas de un archivo .npy, pero ya que bitcoin para añadiendo valores cada dia probablemente en el momento que estes leyendo este texto los valores que hayan añadido seran mayores a los que estan guardados en el archivo .npy que se llama "predicciones.npy". Para solucionar este problema que probablemente hara que bote un error por hacer comparacion de 2 arrays con diferente cantidad de valores lo unico que hay q hacer es descomentar una linea que dice exactamente:

#y_pred = model.predict(X_test)

#np.save('predicciones.npy', y_pred)

Descomentarlo hara que el modelo vuelva a calcular los valores y solucione ese problema.
Tambien se utilizo de kaggle otra base de datos para probar el modelo, el codigo esta en EtheriumTest.
