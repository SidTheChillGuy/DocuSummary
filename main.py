try:
    import gradio as gr
except:
    raise ValueError("please perform action - pip install gradio")
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import chromadb
import os
import subprocess

os.makedirs("tmp", exist_ok=True)
os.makedirs("db/chroma_store", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("src", exist_ok=True)

modelsimported = False

with open("src/userhist.txt", "w") as f:
    f.write("")

theme = gr.themes.Monochrome(
    primary_hue="rose",
    secondary_hue="amber",
    text_size="lg",
    radius_size="sm",
    font=[gr.themes.GoogleFont('Trebuchet MS'), gr.themes.GoogleFont('Lora'), 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'Aptos Mono', 'Consolas', 'monospace'],
).set(
    body_background_fill='*neutral_100',
    body_background_fill_dark='*neutral_950',
    body_text_color='*secondary_950',
    body_text_color_dark='*primary_100',
    body_text_size='*text_sm',
    body_text_color_subdued='*neutral_500',
    body_text_color_subdued_dark='*neutral_500',
    background_fill_primary='*primary_50',
    background_fill_primary_dark='*neutral_700',
    background_fill_secondary='*primary_200',
    background_fill_secondary_dark='*neutral_900',
    border_color_accent='*primary_400',
    border_color_accent_dark='*neutral_900',
    border_color_primary_dark='*neutral_500',
    color_accent_soft='*neutral_400',
    color_accent_soft_dark='*primary_950',
    chatbot_text_size='*text_md',
    table_odd_background_fill='*neutral_200',
    table_odd_background_fill_dark='*neutral_800',
    button_transform_hover='TranslateY(3px);',
    button_transform_active='TranslateY(10px)',
    button_transition='all 0.25s ease'
)

def insdepfn():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
def gobacktomainfn():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
def docvecbuttonfn():
    for i in os.listdir("data"):
        os.remove(f"data/{i}")
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
def ragmainshow():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
def save_files(files):
    import shutil, os
    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("data", filename)
        shutil.copy(file.name, dest_path)
        saved_paths.append(dest_path)
    return f"Saved files:\n" + "\n".join(saved_paths)
def rundocvecprocess():
    yield("starting Document Loading.")
    command = ['python','scripts/cnvpdf2img.py']
    out =''
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=10) as process:
        for line in process.stdout:
            out = out+line
            yield (out)
        process.wait()
        if process.returncode != 0:
            yield(f"Error in first process: {process.stderr.read()}")
            return
        else:
            process.kill()
    yield("\nStarting Step 2: Images to JSON")
    command = ['python','scripts/img2json.py']
    out = ''
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=10) as process:
        for line in process.stdout:
            out = out+line
            yield (out)
        process.wait()
        if process.returncode != 0:
            yield(f"Error in first process: {process.stderr.read()}")
            return
        else:
            process.kill()
    yield("\nStarting Step 3: JSON to Embeddings")
    command = ['python','scripts/json2embed.py']
    out = ''
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=10) as process:
        for line in process.stdout:
            out = out+line
            yield (out)
        process.wait()
        if process.returncode != 0:
            yield(f"Error in first process: {process.stderr.read()}")
            return
        else:
            process.kill()
    yield("\nAll processing steps completed successfully")
def loadmodels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",model_kwargs={"attn_implementation": "flash_attention_2", "device_map": device, "torch_dtype":torch.float16, "quantization_config":quantization_config},tokenizer_kwargs={"padding_side": "left"})
    instructmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ", torch_dtype = torch.float16, device_map=device)
    instructtoken = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ")        
    return embedder, instructmodel, instructtoken
chroma_client = chromadb.PersistentClient(path="db/chroma_store")
collection = chroma_client.get_or_create_collection("my_docs")
embedder, instructmodel, instructtoken = None, None, None
def insdep_pip():
    import subprocess
    command = ["pip", "install","-U","accelerate>=1.8.1", "autoawq>=0.2.9", "bitsandbytes>=0.46.1", "chromadb>=1.0.15", "datasets>=3.6.0", "decord>=0.6.0", "einops>=0.8.1", "executing>=2.2.0", "flash_attn>=2.8.0.post2", "grpcio>=1.73.1", "hf-xet>=1.1.5", "huggingface-hub[hf_xet]>=0.33.2", "idna>=3.10", "jedi>=0.19.2", "Jinja2>=3.1.6", "joblib>=1.5.1", "jsonlines>=4.0.0", "jsonschema>=4.24.0", "jsonschema-specifications>=2025.4.1", "kubernetes>=33.1.0", "llvmlite>=0.44.0", "markdown-it-py>=3.0.0", "ml_dtypes>=0.5.1", "mmh3>=5.1.0", "mpmath>=1.3.0", "numpy>=2.2.6", "nvidia-cublas-cu12>=12.6.4.1", "nvidia-cuda-cupti-cu12>=12.6.80", "nvidia-cuda-nvrtc-cu12>=12.6.77", "nvidia-cuda-runtime-cu12>=12.6.77", "nvidia-cudnn-cu12>=9.5.1.17", "nvidia-cufft-cu12>=11.3.0.4", "nvidia-cufile-cu12>=1.11.1.6", "nvidia-curand-cu12>=10.3.7.77", "nvidia-cusolver-cu12>=11.7.1.2", "nvidia-cusparse-cu12>=12.5.4.2", "nvidia-cusparselt-cu12>=0.6.3", "nvidia-nccl-cu12>=2.26.2", "nvidia-nvjitlink-cu12>=12.6.85", "nvidia-nvtx-cu12>=12.6.77", "onnxruntime>=1.22.0", "pandas>=2.3.0", "parso>=0.8.4", "pdf2image>=1.17.0", "pillow>=11.3.0", "qwen-vl-utils>=0.0.11", "safetensors>=0.5.3", "sentence-transformers>=5.0.0", "timm>=1.0.16", "tokenizers>=0.21.2", "torch>=2.7.1", "torchaudio>=2.7.1", "torchvision>=0.22.1", "tqdm>=4.67.1", "transformers>=4.53.1", "triton>=3.3.1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=10)
    out = ""
    for line in process.stdout:
        out = out+line
        yield (out)
    process.wait()
def answerquery(query,hist):
    print(hist)
    if len(hist)>0:
        with open("src/userhist.txt", "a") as f:
            f.write(str(hist[-2]))
            f.write("\n")
            f.write(str(hist[-1]))
            f.write("\n")
    else:
        with open("src/userhist.txt", "w") as f:
            f.write("")
    global modelsimported
    if modelsimported == False:
        global embedder, instructmodel, instructtoken, prefix_tokens, suffix_tokens, token_false_id, token_true_id
        embedder, instructmodel, instructtoken = loadmodels()
        modelsimported = True
    with torch.no_grad():
        query_embedding = embedder.encode(query)
        top_k = 7
        query_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k)
        retrieved_chunks = query_results["documents"][0]
        messages = [
            {
                "role": "system",
                "content": f"You are Qwen, a helpful assistant who answers queries of the users. Here are some documents that you may use to answer:\n{"\n".join(i for i in retrieved_chunks)}\n\nThe query provided by the user is as follows:\n{query}"
            }
        ]
        text = instructtoken.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = instructtoken([text], return_tensors="pt").to(instructmodel.device)
        generated_ids = instructmodel.generate(
            **model_inputs,
            max_new_tokens=8192
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = instructtoken.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
    torch.cuda.empty_cache()
    return response

with gr.Blocks(theme=theme, title="DocuSummary") as demo:
    # title first page
    with gr.Column(visible=True) as title_screen:
        gr.Markdown("# <center> Welcome to DocuSummary")
        gr.Markdown("""This is a RAG application based on Generative AI models.
                    My goal was to create a simple RAG Pipeline that runs locally, uses available resources efficiently while also serving the purpose of being accurate enough to be utilized on real world applications.
                    
                    For this project, I am using Three GenAI models:
                    1. OpenGVLab's InternVL3 for Image Description Capture.
                    2. Qwen 3 Embedding Model for converting textual Chunks to Vectorized Embeddings.
                    3. Qwen 2.5 Instruct for Query Answering and Generation.
                    
                    I hope you like the project. For any issues, reach out to this [GitHub](https://github.com/SidTheChillGuy/DocuSummary) repository.""", line_breaks=True, container=True)
        with gr.Column():
            gr.Markdown("""<div style="display: flex; justify-content: center; align-items: center;">
                <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="400">
                </div>""", container=False)
            gr.Markdown("""<div style="display: flex; justify-content: center; align-items: center;">
                <img width="400" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
                </div>""", container=False)
        with gr.Row(visible=True):
            insdep = gr.Button(variant="secondary", value="Install the dependencies.",)
            querag = gr.Button(variant="primary", value="Query from your Documents.")
            docvec = gr.Button(variant="secondary", value="Update documents in database.")
    
    # make dependency installer
    with gr.Column(visible=False) as insdep_screen:
        gr.Markdown("# <center>Installing Dependencies")
        gr.Markdown("""This function helps to install dependencies in the project, but its ***recommended to manually install the dependencies instead.***""")
        with gr.Accordion("Dependencies:", open=False):
            gr.Markdown('''```pip install -U "accelerate>=1.8.1", "autoawq>=0.2.9", "bitsandbytes>=0.46.1", "chromadb>=1.0.15", "datasets>=3.6.0", "decord>=0.6.0", "einops>=0.8.1", "executing>=2.2.0", "flash_attn>=2.8.0.post2", "grpcio>=1.73.1", "hf-xet>=1.1.5", "huggingface-hub>=0.33.2", "idna>=3.10", "jedi>=0.19.2", "Jinja2>=3.1.6", "joblib>=1.5.1", "jsonlines>=4.0.0", "jsonschema>=4.24.0", "jsonschema-specifications>=2025.4.1", "kubernetes>=33.1.0", "llvmlite>=0.44.0", "markdown-it-py>=3.0.0", "ml_dtypes>=0.5.1", "mmh3>=5.1.0", "mpmath>=1.3.0", "numpy>=2.2.6", "nvidia-cublas-cu12>=12.6.4.1", "nvidia-cuda-cupti-cu12>=12.6.80", "nvidia-cuda-nvrtc-cu12>=12.6.77", "nvidia-cuda-runtime-cu12>=12.6.77", "nvidia-cudnn-cu12>=9.5.1.17", "nvidia-cufft-cu12>=11.3.0.4", "nvidia-cufile-cu12>=1.11.1.6", "nvidia-curand-cu12>=10.3.7.77", "nvidia-cusolver-cu12>=11.7.1.2", "nvidia-cusparse-cu12>=12.5.4.2", "nvidia-cusparselt-cu12>=0.6.3", "nvidia-nccl-cu12>=2.26.2", "nvidia-nvjitlink-cu12>=12.6.85", "nvidia-nvtx-cu12>=12.6.77", "onnxruntime>=1.22.0", "pandas>=2.3.0", "parso>=0.8.4", "pdf2image>=1.17.0", "pillow>=11.3.0", "qwen-vl-utils>=0.0.11", "safetensors>=0.5.3", "sentence-transformers>=5.0.0", "timm>=1.0.16", "tokenizers>=0.21.2", "torch>=2.7.1", "torchaudio>=2.7.1", "torchvision>=0.22.1", "tqdm>=4.67.1", "transformers>=4.53.1", "triton>=3.3.1"``` ''',line_breaks=True)
            gr.Markdown("Additionally, Install Poppler-Utils for your respective system and make sure it is accessible.")
            with gr.Accordion(label="AutoRun", open=False):
                insdep_lazypip = gr.Button("Install it for me")
                with gr.Row():
                    insdep_lazypip_text = gr.Text(label="Console Out", max_lines=10, interactive=False)
        with gr.Accordion("Models:", open=False):
            gr.Markdown("""1. OpenGVLab's InternVL3""",line_breaks=True)
            with gr.Accordion(label="Python Code", open=False):
                gr.Markdown("""```
                            from huggingface_hub import snapshot_download
                            snapshot_download(repo_id='OpenGVLab/InternVL3-2B', repo_type='model')""", line_breaks=True)

            gr.Markdown("""2. Qwen 3 Embeddings""",line_breaks=True)
            with gr.Accordion(label="Python Code", open=False):
                gr.Markdown("""```
                            from huggingface_hub import snapshot_download
                            snapshot_download(repo_id='Qwen/Qwen3-Embedding-0.6B', repo_type='model')""", line_breaks=True)

            gr.Markdown("""3. Qwen 2.5 Instruct""",line_breaks=True)
            with gr.Accordion(label="Python Code", open=False):
                gr.Markdown("""```
                            from huggingface_hub import snapshot_download
                            snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct-AWQ', repo_type='model')""", line_breaks=True)
        gobacktomain_insdep = gr.Button("Back to Main")
    
    # make upload docs screen
    with gr.Column(visible=False) as upload_cnv:
        gr.Markdown("# Upload your documents to start embedding.")
        file_upload = gr.File(label="Upload multiple files", file_types=["file"], interactive=True, file_count="multiple")
        output = gr.Textbox(label="Saved File Paths", lines=5)
        file_upload.upload(save_files, inputs=file_upload, outputs=output)
        startprocessingdoc = gr.Button("Start Document Processing.")
        processdocout = gr.Textbox(label="Process output", placeholder="The process takes time to run.")
        gobacktomain_docvec = gr.Button("Back to Main")
    
    # query rag
    with gr.Column(visible=False) as ragmain:
        gr.Markdown("""# <center>DocuSummary Chat""")
        gr.ChatInterface(fn=answerquery, type="messages",chatbot=gr.Chatbot(height=450, show_copy_button=True, type="messages"),
                        textbox=gr.Textbox(placeholder="Start chatting", container=True, max_lines=5))
        with gr.Row():
            ragtomain = gr.Button("Back to Main Screen")
            downloadhist = gr.DownloadButton(label="Download user chat history", value="src/userhist.txt")
    
    # button functions
    insdep.click(fn=insdepfn, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])
    gobacktomain_insdep.click(fn=gobacktomainfn, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])
    insdep_lazypip.click(fn=insdep_pip, inputs=[], outputs=[insdep_lazypip_text], show_progress='full')
    docvec.click(fn=docvecbuttonfn, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])
    gobacktomain_docvec.click(fn=gobacktomainfn, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])
    startprocessingdoc.click(fn=rundocvecprocess, inputs=[], outputs=[processdocout])
    querag.click(fn=ragmainshow, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])
    ragtomain.click(fn=gobacktomainfn, inputs=[], outputs=[title_screen, insdep_screen, upload_cnv, ragmain])

demo.launch()