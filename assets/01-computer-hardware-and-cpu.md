# Section 1: Computer Hardware and CPU Programming

## Motivation

Most of us started by building a PC for gaming — picking components without truly understanding how they work. As a Computer Science student, curiosity drives you to ask: **how does my computer actually work?**

## Concurrency vs Parallelism

- **Concurrency**: Overlaps tasks to hide idle time (single core, interleaved execution)
- **Parallelism**: Runs tasks simultaneously across multiple cores

## Threads and Multi-threading

A process consists of multiple **threads** — each a logical sequential execution unit. Multi-threading splits a large task (e.g., video rendering) into smaller threads assigned to separate CPU cores for parallel execution.

## Why Traditional CPU Computation is Slow

The CPU constantly fetches data from RAM, computes with the **ALU (Arithmetic Logic Unit)**, and transfers results via the **Control Unit**. However, the CPU isn't dedicated to a single task — it juggles dozens of processes (Chrome tabs, Spotify, OS services). The ALU is designed as a general-purpose swiss army knife, optimized for complex control flow rather than raw throughput.

## CPU Execution Model — Flynn's Taxonomy

Michael J. Flynn's taxonomy classifies execution models:

| Model | Description |
|-------|-------------|
| SISD | Single Instruction, Single Data |
| SIMD | Single Instruction, Multiple Data |
| MISD | Multiple Instruction, Single Data |
| MIMD | Multiple Instruction, Multiple Data |

CPUs typically operate in MIMD — multiple instructions on multiple data streams across cores.

## Why GPU?

GPUs outperform CPUs for parallel workloads because they trade complex control logic for **massive parallelism** — thousands of simple cores executing the same instruction on different data simultaneously.

## Disclaimer

GPU performance depends on many factors beyond architecture:

- PCIe & RAM data transfer bottlenecks
- GPU interconnection between nodes/clusters
- Thermal limits
- Numerical precision requirements
- GPU generation and firmware
- Input data scale

This document focuses on the **core architecture**, why GPUs excel at high-performance computing, and how to write custom kernels.
