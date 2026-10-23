"""
The acquisition-steps preview shown at Start is saved as acquisition_steps.txt
at the top level of the run's experiment folder (next to the sample images and
the spectra/ + metadata/ subfolders), once per run.
"""

import os

from raman_control.gui.threads.acquisition_thread import (
    AcquisitionContext, AcquisitionWorker)


def make_worker(tmp_path, **params):
    ctx = AcquisitionContext(motion=None, spectrometer=None, nanodrive=None,
                             frame_provider=lambda: None,
                             position_provider=lambda: (0.0, 0.0, 0.0),
                             wavelengths_nm_provider=lambda: None)
    return AcquisitionWorker(ctx, [], {'save_path': str(tmp_path), **params})


def test_preview_saved_in_experiment_folder(tmp_path):
    w = make_worker(tmp_path, steps_preview="Setup:\n  - Move to point 1\n")
    spectra_dir, _ = w._experiment_dirs()
    exp_dir = os.path.dirname(spectra_dir)
    path = os.path.join(exp_dir, AcquisitionWorker.STEPS_FILENAME)
    with open(path, encoding='utf-8') as f:
        text = f.read()
    assert text.startswith("Acquisition steps preview (run started ")
    assert "Setup:\n  - Move to point 1\n" in text


def test_preview_written_once_per_run(tmp_path):
    w = make_worker(tmp_path, steps_preview="first")
    spectra_dir, _ = w._experiment_dirs()
    path = os.path.join(os.path.dirname(spectra_dir),
                        AcquisitionWorker.STEPS_FILENAME)
    os.remove(path)
    w._experiment_dirs()  # later saves in the same run don't rewrite it
    assert not os.path.exists(path)


def test_no_preview_no_file(tmp_path):
    w = make_worker(tmp_path)
    spectra_dir, _ = w._experiment_dirs()
    assert not os.path.exists(os.path.join(
        os.path.dirname(spectra_dir), AcquisitionWorker.STEPS_FILENAME))


def test_no_save_path_no_folder(tmp_path):
    w = make_worker(tmp_path, steps_preview="x")
    w._p['save_path'] = ''
    assert w._experiment_dirs() == (None, None)
    assert os.listdir(tmp_path) == []
