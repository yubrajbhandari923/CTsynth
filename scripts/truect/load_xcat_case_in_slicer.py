"""Load one XCAT pipeline case in 3D Slicer: simulated CT as background, organ labels as an overlay (adjustable opacity).

In Slicer's Python console (View > Python Interactor):
    CASE_DIR = "/path/to/xcat_pipeline/copd_22728"      # the folder that contains nifti/
    exec(open("/path/to/load_xcat_case_in_slicer.py").read())
Or:  Slicer --python-script load_xcat_case_in_slicer.py  (CASE_DIR then comes from the environment variable XCAT_CASE_DIR).
NOT TESTED inside Slicer by the author (no Slicer in the build environment); the fallback is File > Add Data, see the README.
Orientation letters are PROVISIONAL: check R/L/A/P/S/I. All files in nifti/ ending *_recon_grid / reconstructed / ideal share one grid.
"""
import os

import slicer

CASE_DIR = globals().get("CASE_DIR") or os.environ.get("XCAT_CASE_DIR") or "."
nii = os.path.join(CASE_DIR, "nifti")
ct = slicer.util.loadVolume(os.path.join(nii, "reconstructed_hu70.nii.gz"))
ideal = slicer.util.loadVolume(os.path.join(nii, "phantom_ideal_hu70.nii.gz"))
lab = slicer.util.loadLabelVolume(os.path.join(nii, "phantom_organ_labels_recon_grid.nii.gz"))
mat = slicer.util.loadLabelVolume(os.path.join(nii, "phantom_labels_recon_grid.nii.gz"))
for v, name in ((ct, "simulated_CT_hu70"), (ideal, "ideal_CT_hu70"), (lab, "organ_labels"), (mat, "material_labels")):
    v.SetName(name)
ctbl = slicer.util.loadColorTable(os.path.join(nii, "organ_labels.ctbl"))
lab.GetDisplayNode().SetAndObserveColorNodeID(ctbl.GetID())
mctbl = slicer.util.loadColorTable(os.path.join(nii, "material_labels.ctbl"))
mat.GetDisplayNode().SetAndObserveColorNodeID(mctbl.GetID())
for v in (ct, ideal):
    d = v.GetDisplayNode(); d.SetAutoWindowLevel(False); d.SetWindowLevel(1400, -300)   # lung window; use W=400, L=40 for soft tissue
# background = simulated CT, label = organ labels at 40% opacity; set label opacity with the slider in the slice controller
slicer.util.setSliceViewerLayers(background=ct, foreground=ideal, label=lab, foregroundOpacity=0.0, labelOpacity=0.4, fit=True)
# optional: segmentation node (per-organ visibility/opacity, 3D surfaces)
seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "organ_segmentation")
seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg)
seg.GetDisplayNode().SetOpacity(0.4)
print("loaded", CASE_DIR, "; check the R/L/A/P/S/I letters (PROVISIONAL)")
