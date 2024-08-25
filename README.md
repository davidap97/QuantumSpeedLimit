# QuantumSpeedLimit
The notebooks show how to estimate the quantum speed limit of a unitary gate using Krotov's method of optimization in a so-called minimal time approach.


The Python script "sample_job_final.py" outlines a quantum speed limit search that is entirely written in Python code and makes use of parallelisation.

Moreover, the Jupyter notebook "QSLsearch_EntanglingGates" elucidates the methodology for conducting a quantum speed limit search through parallelisation, utilising the QDYNpylib library. Additionally, it furnishes elucidations pertaining to the three-qubit system.

The subsequent application of the "Infidelity_Curve" notebook permits the observation of the quantum speed limit candidate. Further details are provided in that notebook.

The "Test_Discretization" notebook provides a step-by-step guide on how to assess the "physical magnitude" of an optimized pulse. This ultimately determines whether the results correspond to actual physical phenomena.


In "Do_Runfolders" it is written how to perform a quantum speed limit search using the power of cluster nodes. The execution of that notebook requires the subsequent execution of the "create_batch_script" notebook.


