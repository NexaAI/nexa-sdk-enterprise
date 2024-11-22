from nexa_enterprise.gguf.nexa_inference_text import NexaTextInference
from nexa_enterprise.gguf.llama.kv_cache import run_inference_with_disk_cache
import os
import shutil
from nexa_enterprise.gguf.lib_utils import is_gpu_available
from nexa_enterprise.gguf.llama.llama import Llama
from nexa_enterprise.general import pull_model

# Remove cache directories if they exist
if os.path.exists("llama.cache"):
    shutil.rmtree("llama.cache")
if os.path.exists("llama2.cache"):
    shutil.rmtree("llama2.cache")

base_prompt = """NVIDIA, founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, has grown from a modest startup into a global leader in the graphics processing unit (GPU) industry, fundamentally transforming the landscape of computing, gaming, artificial intelligence, and various other technological sectors. Initially, NVIDIA focused on developing high-performance graphics cards for the burgeoning PC gaming market, with their early RIVA and GeForce series setting new standards for visual fidelity and processing speed, thereby enabling gamers to experience more immersive and visually stunning environments. The introduction of the GeForce 256 in 1999, marketed as the "world's first GPU," marked a significant milestone by integrating transform and lighting calculations directly on the GPU, offloading these tasks from the central processing unit (CPU) and thereby enhancing overall system performance and graphical realism. As the demand for more sophisticated graphics and higher resolution displays grew, NVIDIA continued to innovate with successive generations of GPUs, each offering substantial improvements in processing power, memory bandwidth, and energy efficiency, which not only catered to gamers but also appealed to professionals in fields such as video editing, 3D rendering, and scientific visualization, where the ability to handle complex computations and large datasets is crucial. Beyond gaming and professional graphics, NVIDIA recognized the potential of GPUs in accelerating a wide array of computational tasks, leading to the development of CUDA (Compute Unified Device Architecture) in 2006, a parallel computing platform and programming model that unlocked the GPU's processing capabilities for general-purpose computing, thereby revolutionizing industries like artificial intelligence, machine learning, and deep learning by providing the computational horsepower necessary to train sophisticated neural networks and process vast amounts of data at unprecedented speeds. This strategic pivot positioned NVIDIA at the forefront of the AI revolution, with their GPUs becoming the backbone of many AI frameworks and applications, from autonomous vehicles and robotics to healthcare diagnostics and natural language processing, enabling breakthroughs that were previously unattainable with traditional CPU-based systems. The company's commitment to innovation is further exemplified by their development of specialized hardware, such as the Tensor Cores introduced in the Volta and subsequent architectures, which are designed specifically to accelerate tensor operations that are fundamental to deep learning algorithms, significantly enhancing the efficiency and performance of AI models. Additionally, NVIDIA has made substantial investments in software ecosystems, creating tools like NVIDIA Deep Learning SDK, cuDNN, and TensorRT, which provide developers with the resources needed to optimize and deploy AI applications seamlessly across various platforms, including data centers, edge devices, and cloud environments. The expansion into data centers with the acquisition of Mellanox Technologies in 2020 underscored NVIDIA's vision of providing comprehensive solutions that encompass not only processing power but also high-speed networking and interconnect technologies essential for large-scale AI deployments and high-performance computing (HPC) workloads. Moreover, NVIDIA's foray into the automotive industry through their DRIVE platform has paved the way for advancements in autonomous driving technology, offering integrated hardware and software solutions that enable vehicles to perceive their environment, make real-time decisions, and navigate safely, thus contributing to the future of smart transportation systems. The company's influence extends into the realms of virtual reality (VR) and augmented reality (AR), where NVIDIA GPUs power the rendering and real-time processing required to create seamless and interactive virtual experiences, bridging the gap between digital and physical realities and opening new possibilities in entertainment, education, and professional training. NVIDIA's dedication to sustainability and energy efficiency is evident in their architectural advancements, such as the adoption of smaller process nodes and the implementation of power-saving technologies that reduce the environmental footprint of their products without compromising performance, reflecting a broader commitment to responsible innovation in an era where the demand for computational power continues to escalate. Furthermore, NVIDIA has cultivated a robust ecosystem through strategic partnerships with leading technology companies, research institutions, and developers worldwide, fostering collaboration and driving the adoption of their technologies across diverse applications and industries, thereby amplifying their impact and ensuring that their GPUs remain integral to the advancement of cutting-edge technologies. The company's venture into the burgeoning field of quantum computing, through research and development initiatives aimed at integrating quantum processors with classical GPU architectures, signifies NVIDIA's forward-thinking approach and dedication to staying at the forefront of technological evolution, preparing for a future where hybrid computing systems leverage the strengths of both quantum and classical paradigms to solve complex problems that are currently beyond reach. Additionally, NVIDIA's emphasis on education and community engagement, demonstrated through initiatives like the NVIDIA Deep Learning Institute and various developer programs, has empowered countless individuals and organizations to harness the full potential of GPU computing, democratizing access to advanced technologies and fostering a culture of continuous learning and innovation. As the digital landscape evolves, NVIDIA remains a pivotal force, driving progress through relentless innovation, strategic foresight, and a commitment to excellence, ensuring that their GPUs not only meet the current demands of high-performance computing but also anticipate and shape the future needs of a rapidly advancing technological world, where the convergence of AI, machine learning, VR/AR, autonomous systems, and quantum computing will redefine the boundaries of what is possible, and NVIDIA's GPUs will continue to be the critical enablers of these transformative advancements, solidifying the company's legacy as a cornerstone of modern computing and a catalyst for the next wave of technological breakthroughs that will shape the future of industries, economies, and societies worldwide."""
base_prompt2 = """Nexa AI, a Cupertino-based generative AI company, is redefining on-device multimodal intelligence with its Octopus model and SDK, offering state-of-the-art solutions for text, audio, and vision tasks. Leveraging proprietary innovations like functional token training and a decoder-decoder architecture for long-context handling, Nexa enables businesses to deploy high-performance AI systems on edge devices with unparalleled efficiency. By partnering with hardware giants like AMD and collaborating with enterprise leaders such as HP, Nexa ensures its technology meets the stringent demands of latency, power efficiency, and multimodal versatility, making it a critical player in the future of AI-driven enterprise applications."""

# === prefilling cache ===
# Test both with and without cache

model_local_path, model_type = pull_model("llama3")

model = Llama(
    model_path=model_local_path,
    n_ctx=8096,
    n_gpu_layers=-1,
    chat_format="llama3",
    verbose=True,
)

run_inference_with_disk_cache(
    model=model,
    cache_prompt=base_prompt,
    total_prompt=base_prompt,
    max_tokens=1,
    use_cache=True,
    cache_dir="llama.cache",
)

run_inference_with_disk_cache(
    model=model,
    cache_prompt=base_prompt2,
    total_prompt=base_prompt2,
    max_tokens=1,
    use_cache=True,
    cache_dir="llama2.cache",
)

# === inference ===
print("=== 1st Inference ===")
extended_prompt = (
    base_prompt
    + " Summarize this article and return me a summary of it in 30 words or less."
)
output = run_inference_with_disk_cache(
    model=model,
    cache_prompt=base_prompt,
    total_prompt=extended_prompt,
    max_tokens=256,
    use_cache=True,
    cache_dir="llama.cache",
)
for chunk in output:
    if "choices" in chunk:
        print(chunk["choices"][0]["text"], end="", flush=True)

print("=== 2nd Inference ===")
extended_prompt2 = (
    base_prompt2
    + " Summarize this article and return me a summary of it in 30 words or less."
)
output = run_inference_with_disk_cache(
    model=model,
    cache_prompt=base_prompt2,
    total_prompt=extended_prompt2,
    max_tokens=256,
    use_cache=True,
    cache_dir="llama2.cache",
)
for chunk in output:
    if "choices" in chunk:
        print(chunk["choices"][0]["text"], end="", flush=True)
