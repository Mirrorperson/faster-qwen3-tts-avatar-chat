import grpc
import sys
import os
import time
import tempfile
from concurrent import futures

# Add current path for protobuf imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import avatar_pb2
import avatar_pb2_grpc

# We must run from the FasterLivePortrait root directory so all relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline


class AvatarServiceServicer(avatar_pb2_grpc.AvatarServiceServicer):
    def __init__(self):
        print("Loading FasterLivePortrait pipeline into memory (TensorRT)...")
        infer_cfg = OmegaConf.load("configs/trt_infer.yaml")
        infer_cfg.infer_params.flag_pasteback = True
        self.pipe = GradioLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
        print("Pipeline loaded! Server ready.")

    def AnimateAvatar(self, request, context):
        print(f"Received request: Image={len(request.source_image)}b, Audio={len(request.driving_audio)}b")
        
        img_path = None
        aud_path = None
        final_output_path = None

        try:
            # 1. Save incoming bytes to temporary files
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_img:
                f_img.write(request.source_image)
                img_path = f_img.name
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_aud:
                f_aud.write(request.driving_audio)
                aud_path = f_aud.name

            print(f"Saved temp files: Image={img_path}, Audio={aud_path}")
            
            # 2. Run the audio-driven portrait animation using JoyVASA
            # run_audio_driving() uses JoyVASA to convert the WAV into facial motion,
            # then renders the animated portrait video and mixes the audio in.
            print("Running JoyVASA audio-driven animation...")
            start_time = time.time()

            vsave_org_path, vsave_crop_path, total_time = self.pipe.run_audio_driving(
                driving_audio_path=aud_path,
                source_path=img_path
            )

            print(f"Animation finished in {total_time:.2f} seconds.")
            print(f"Output video (org): {vsave_org_path}")

            # 3. Use the full-frame output (org), not the cropped version
            final_output_path = vsave_org_path

            # 4. Read the output video bytes
            with open(final_output_path, "rb") as f_vid:
                video_bytes = f_vid.read()

            print(f"Streaming back MP4: {len(video_bytes)} bytes")
            return avatar_pb2.AvatarResponse(video_output=video_bytes)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return avatar_pb2.AvatarResponse()
        finally:
            # Clean up temp input files
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            if aud_path and os.path.exists(aud_path):
                os.remove(aud_path)


def serve():
    GRPC_OPTIONS = [
        ('grpc.max_send_message_length', -1),    # unlimited
        ('grpc.max_receive_message_length', -1), # unlimited
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=GRPC_OPTIONS)
    avatar_pb2_grpc.add_AvatarServiceServicer_to_server(AvatarServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("FasterLivePortrait gRPC Server is listening on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
