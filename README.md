# SurfaceRegistration


Functions are in registration.py and utils.py

-bodyFatSofttissueMask.py contains the code for segmentation of the objects from CT. Surfaces are generated from these segmentations using nii2mesh.
-Main registration function is in registration.py. run.py calls this function
-Rest of scripts call preprocesing and postprocessing functions that are found in registration.py and utils.py
