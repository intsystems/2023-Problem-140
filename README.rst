|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Адаптация архитектуры модели глубокого обучения с контролем эксплуатационных характеристик
    :Тип научной работы: M1P
    :Автор: Савелий Бородин
    :Научный руководитель: Олег Бахтеев
    :Научный консультант: Константин Яковлев

Abstract
========

The paper investigates the problem of structural pruning with respect to target hardware properties. A method considers performance of a target platform to optimize both accuracy and latency on the platform. We use a hypernetwork to generate a pruned model for a desired trade-off between accuracy and latency. The hypernetwork is trained end-to-end with backpropagation through the main model. The model adapts to benefits and weaknesses of hardware, which is especially important for mobile devices with limited computation budget. To evaluate the performance of the proposed algorithm, we conduct experiments on the CIFAR-10 dataset with ResNet18 as a backbone-model using different hardware and compare the resulting architectures with architectures obtained by greedy algorithm.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/2023-Problem-140/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.com/intsystems/2023-Problem-140/blob/master/code/basic_experiment.ipynb>`_ and `here <https://github.com/intsystems/2023-Problem-140/blob/master/code/main_experiment.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/2023-Problem-140/blob/master/code/basic_experiment.ipynb>`_ and `colab <http://colab.research.google.com/github/intsystems/2023-Problem-140/blob/master/code/main_experiment.ipynb>`_.
