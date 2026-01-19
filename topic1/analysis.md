Part 4

Looking at the results of the 3 runs, we can see that GPU was fastest, followed by CPU with no quantization, and lastly CPU with 4 bit quantization, with the GPU being almost 3 times as fast as the CPU. Unexpectedly, quantization actually made the CPU slower. Looking up potential causes, this seemes to be because standard CPUs do not have hardware acceleration for 4-bit math, meaning the CPU has to dequantize the weights, resulting in added overhead.

Part 6/7

The mistakes do not appear to be random. For example, almost all models struggle on more complex questions like math and physics based questions. In addition, for the majority of questions where stronger models (like Qwen 7B) fail, the weaker models also fail (like Llama 1B). In short, the models with less parameters tend to perform worse overall, while models with more parameters tend to do better.
