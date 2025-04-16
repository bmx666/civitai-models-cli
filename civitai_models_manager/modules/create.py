import os
from enum import Enum
from typing import Dict, List, Final, Optional
from questionary import select
from questionary import text as prompt

from rich import print_json
from rich.console import Console
import typer

from civitai_models_manager.modules.helpers import feedback_message
from .details import get_model_details, process_model_data

os.environ["CIVITAI_API_TOKEN"] = os.getenv("CIVITAI_TOKEN")
import civitai

console = Console()

SCHEDULERS: Final[List[str]] = [
    "EulerA",
    "Euler",
    "LMS",
    "Heun",
    "DPM2",
    "DPM2A",
    "DPM2SA",
    "DPM2M",
    "DPMSDE",
    "DPMFast",
    "DPMAdaptive",
    "LMSKarras",
    "DPM2Karras",
    "DPM2AKarras",
    "DPM2SAKarras",
    "DPM2MKarras",
    "DPMSDEKarras",
    "DDIM",
    "PLMS",
    "UniPC",
    "Undefined",
    "LCM",
    "DDPM",
    "DEIS",
]


class CreateOptions(Enum):
    MODEL = "What model would you like to use? [Required]"
    PROMPT = "Positive Prompt [Required]"
    NEGATIVE = "Negative Prompt [Optional]"
    SCHEDULER = "Scheduler [Optional]"
    STEPS = "Steps [Optional]"
    CFG_SCALE = "CFG Scale [Optional]"
    WIDTH_HEIGHT = "Width x Height [Required]"
    CANCEL = "Cancel"


create_group = typer.Typer()


def get_lora_details(CIVITAI_MODELS, CIVITAI_VERSIONS,lora_id: int) -> Optional[Dict]:
    try:
        lora_data = get_model_details(
            CIVITAI_MODELS, 
            CIVITAI_VERSIONS,
            lora_id,
        )
        if lora_data and lora_data.get("type") == "LORA":
            return lora_data
        else:
            feedback_message(f"Model with ID {lora_id} is not a LoRA.", "error")
            return None
    except Exception as e:
        feedback_message(f"Error fetching LoRA details: {str(e)}", "error")
        return None


def generate_image(
    CIVITAI_MODELS,
    CIVITAI_VERSIONS,
    air: str,
    pos_prompt: str,
    neg_prompt: str,
    width: int,
    height: int,
    scheduler: str,
    steps: int,
    cfg_scale: float,
    lora_list: List[int] = []
):
    """Generate the image using the Civitai SDK."""
    feedback_message("Generating the image...", "info")
    try:
        input_data = {
            "model": air,
            "params": {
                "prompt": pos_prompt,
                "negativePrompt": neg_prompt,
                "scheduler": scheduler,
                "steps": steps,
                "cfgScale": cfg_scale,
                "width": width,
                "height": height,
                "seed": -1,
                "clipSkip": 1
            },
        }

        if lora_list:
            input_data["additionalNetworks"] = {}
            feedback_message(f"Processing {len(lora_list)} LoRA models...", "info")
            for lora_id in lora_list:
                lora_data = get_lora_details(lora_id, CIVITAI_MODELS, CIVITAI_VERSIONS)
                if lora_data and 'versions' in lora_data and lora_data['versions']:
                    lora_air = lora_data["versions"][0].get("air")
                    if lora_air:
                        input_data["additionalNetworks"][lora_air] = {
                            "type": "Lora",
                            "strength": 1.0,
                        }
                        feedback_message(f"Added LoRA: {lora_air}", "info")
                    else:
                        feedback_message(f"LoRA AIR not found for ID: {lora_id}", "warning")
                else:
                    feedback_message(f"Failed to get details for LoRA ID: {lora_id}", "warning")

        feedback_message("Submitting image generation request...", "info")
        console.print("Input data:", style="bold")
        print_json(data=input_data)

        response = civitai.image.create(input_data)
        feedback_message("Image generation request submitted successfully.", "success")
        return response
    except Exception as e:
        feedback_message(f"Error generating image: {str(e)}", "error")
        return None


def fetch_job_details(job_id: str, user_id: str, detailed: bool) -> None:

    try:
        if job_id:
            job_details = civitai.jobs.get(id=job_id)
            if job_details:
                console.print("Job details:", style="bold")
                print_json(data=job_details)
            else:
                feedback_message("No job details found.", "warning")
        elif user_id:
            query = {"properties": {"userId": user_id}}
            jobs = civitai.jobs.query(detailed=detailed, query_jobs_request=query)
            if jobs:
                console.print("Jobs:", style="bold")
                print_json(data=jobs)
            else:
                feedback_message("No jobs found for the given user ID.", "warning")
        else:
            feedback_message("Please provide either a job ID or a user ID.", "error")
    except Exception as e:
        feedback_message(f"Error fetching job details: {str(e)}", "error")


def cancel_job(job_id: str):
    """Cancel a job by its Job ID."""
    try:
        response = civitai.jobs.cancel(job_id)
        if response:
            console.print("Job cancellation response:", style="bold")
            print_json(data=response)
        else:
            feedback_message("Failed to cancel the job.", "error")
    except Exception as e:
        feedback_message(f"Error cancelling job: {str(e)}", "error")


def create_image_cli(
    CIVITAI_MODELS: str,
    CIVITAI_VERSIONS: str,
    requested_model: int,
    lora_list: List[int],
) -> None:
    try:
        raw_model = get_model_details(CIVITAI_MODELS, CIVITAI_VERSIONS, requested_model)
        processed_model = process_model_data(raw_model)
        if not processed_model:
            feedback_message(f"No model found with ID: {requested_model}", "error")
            return

        selected_model = processed_model if processed_model["versions"] else raw_model

        # Select version if multiple versions exist
        if len(selected_model["versions"]) > 1:
            version_choices = [
                {
                    'name': f"{v['id']} - {v['name']}",
                    'value': v
                } for v in selected_model["versions"]
            ]

            selected_version = select(
                "Select a version",
                choices=version_choices,
            ).ask()
            

            if selected_version:
                arn = selected_version["air"]
            else:
                feedback_message("No version selected. Aborting.", "error")
                return
        else:
            arn = selected_model["versions"][0]["air"]
            

        # # Gather image generation parameters
        pos_prompt = prompt("Enter positive prompt:").ask()
        if not pos_prompt:
            feedback_message("Positive prompt is required.", "error")
            return

        neg_prompt = prompt("Enter negative prompt (optional):").ask()

        width_height = prompt("Enter width x height (e.g., 1024x1024):", default="1024x1024").ask()
        try:
            width, height = map(int, width_height.split("x"))
        except ValueError:
            feedback_message(
                "Invalid width x height format. Please use format like '1024x1024'.",
                "error",
            )
            return

        scheduler = select("Select scheduler:", choices=SCHEDULERS).ask()

        steps = prompt("Enter number of steps:").ask()
        try:
            steps = int(steps)
        except ValueError:
            feedback_message(
                "Invalid number of steps. Please enter an integer.", "error"
            )
            return

        cfg_scale = prompt("Enter CFG scale:").ask()
        try:
            cfg_scale = float(cfg_scale)
        except ValueError:
            feedback_message("Invalid CFG scale. Please enter a number.", "error")
            return

        #Generate image
        response = generate_image(
            CIVITAI_MODELS,
            CIVITAI_VERSIONS,
            arn,
            pos_prompt,
            neg_prompt,
            width,
            height,
            scheduler,
            steps,
            cfg_scale,
            lora_list,
        )

        if response:
            console.print("Image generation response:", style="bold")
            print_json(data=response)

            # Fetch and display job details
            if "jobId" in response:
                job_details = fetch_job_details(response["jobId"])
                if job_details:
                    console.print("Job details:", style="bold")
                    print_json(data=job_details)
            else:
                feedback_message("No job ID found in the response.", "warning")
        else:
            feedback_message("Image generation failed.", "error")

    except Exception as e:
        feedback_message(f"An unexpected error occurred: {str(e)}", "error")
