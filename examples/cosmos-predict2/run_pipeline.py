from diffusers.utils import export_to_video, load_video

from optimum.rbln.diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import RBLNCosmos2_5_PredictBasePipeline


# pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
#     "nvidia/Cosmos-Predict2.5-2B",
#     revision="diffusers/base/pre-trained",
#     export=True,
#     rbln_config={
#         "text_encoder": {
#             "visual": {
#                 "max_seq_len": 2048,
#             },
#             "tensor_parallel_size": 8,
#             # "kvcache_partition_len": 16_384,
#             "max_seq_len": 512,
#         },
#         # "transformer": {
#         #     "num_devices": 8,
#         # },
#         # "safety_checker": {"device": 8, "qwen3guard": {"tensor_parallel_size": 1}},
#     },
# )

# pipe.save_pretrained("cosmos-predict2.5_text_encoder")

pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
    # "cosmos-predict2.5_text_encoder",
    "cosmos-predict2.5_core_compile_te_tm_vae",
    # "cosmos-predict2.5_core_compile_vae_",
    export=False,
    rbln_config={
        "text_encoder": {
            # the visual tower is never called (text-only prompts); skip its runtime
            "visual": {
                "create_runtimes": False,
            },
            "device": [16, 17, 18, 19, 20, 21, 22, 23],
        },
        "transformer": {
            "device": [24, 25, 26, 27, 28, 29, 30, 31],
        },
        "vae": {
            "device_map": {"encoder": 0, "decoder": 1},
        },
    },
)

# input_text = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
# input_text = "In a breathtaking nocturnal tableau, a bustling city bus terminal comes alive under the soft, ethereal glow of overhead lights, where a row of gleaming double-decker buses stands in a harmonious lineup. The central bus, adorned with the number '87D' and a striking red stripe, commands attention, its headlights casting a warm, inviting light that dances across the polished surfaces of its neighbors. The static scene is punctuated by the gentle hum of the terminal, where a few passengers linger, their silhouettes framed by the soft illumination, creating a tranquil yet dynamic atmosphere. As the video unfolds, the central bus begins to stir, its engine purring to life, and the headlights flicker brighter, casting dynamic reflections on the surrounding buses. The subtle shift in the lineup reveals the bustling activity within, as passengers board and disembark, their movements echoing the rhythm of the city. The bus glides forward, creating a momentary gap in the lineup before smoothly coming to a halt, resuming its position in the queue. Above, the illuminated signage in Chinese characters glows softly, enhancing the vibrant urban atmosphere. The camera captures this nocturnal transit hub with a steady gaze, employing a shallow depth of field that blurs the background, inviting viewers to immerse themselves in the quiet beauty of this nocturnal ballet of transportation. The interplay of light and shadow, the gentle hum of the engines, and the serene yet dynamic energy of the scene create a mesmerizing visual symphony that celebrates the essence of urban life under the starlit sky."
# image_path = "bus_terminal.jpg"
# input_image = Image.open(image_path)

# input_text = "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
# input_text = "In a sprawling industrial workshop, bathed in the ethereal glow of welding sparks, a state-of-the-art robotic arm, adorned with a powerful welding torch, gracefully hovers above a colossal metal structure. The camera, positioned at a slightly elevated angle, captures the intricate latticework and robust framework of the structure, where the torch's brilliant blue flame dances, casting mesmerizing light that weaves through the scene. Thick, swirling smoke billows from the welding site, partially obscuring the view yet enhancing the dynamic energy of the workshop, where the air is thick with the scent of molten metal. The background reveals a labyrinth of industrial machinery and ventilation systems, hinting at the bustling activity of a high-tech manufacturing facility. As the welding process unfolds, the metal surface beneath the torch glows a fiery red, revealing the transformative power of the flame, while the camera remains fixed, allowing viewers to immerse themselves in the rhythmic dance of sparks and smoke. Each frame is a masterclass in precision and artistry, showcasing the harmonious marriage of human ingenuity and technological prowess in this cinematic exploration of modern manufacturing. The robotic arm's fluid, deliberate movements are accentuated by dynamic lighting, creating a stunning visual narrative that celebrates the beauty and complexity of industrial craftsmanship. The workshop's vastness is accentuated by the interplay of light and shadow, while the camera's steady gaze invites viewers to appreciate the intricate details of the welding process, from the delicate sparks to the robust machinery that surrounds them."
# image_path = "robot_welding.jpg"
# input_image = Image.open(image_path)

# video = pipe(image=input_image, prompt=input_text, height=704, width=1280).frames[0]

# 220 tokens
input_text = "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
# 316 tokens
input_text = "In a sun-drenched, modern kitchen, bathed in the warm embrace of natural light, a sleek robotic arm, predominantly white with striking black joints and cables, gracefully navigates the air, cradling a light green pitcher brimming with a shimmering, clear liquid. The soft, diffused light filters through expansive windows, casting a golden-hour glow on the pristine glass table, where a clear glass awaits, partially filled with a rich, reddish-brown beverage. To the left, an opened jar reveals a similar, enticing substance, while a white vase adornedwith vibrant flowers and a plush brown couch create a cozy, contemporary ambiance. The robotic arm, equipped with a precision gripper, hovers above the table, its movements fluid and deliberate as it tilts the pitcher, allowing the liquid to cascade into the glass in a mesmerizing dance of light and color. Each drop glistens, reflecting the soft light, as the glass is elegantly filled to the brim, showcasing the robotic arm's unparalleled precision and finesse. The scene remains undisturbed, with a second robotic arm on the right maintaining a steady presence, enhancing the futuristic allure of this automated culinary ballet. The camera captures this harmonious interplay with a static shot, employing shallow depth of field to emphasize the robotic arms' intricate choreography, while dynamic color grading accentuates the rich textures and vibranthues of the kitchen, inviting viewers to immerse themselves in this modern kitchen symphony."
image_path = "robot_pouring.mp4"
input_image = load_video(image_path)
video = pipe(video=input_image, prompt=input_text, height=704, width=1280).frames[0]


export_to_video(video, f"generated_{image_path.split('.')[0]}_upsampled.mp4", fps=16)
print(f"Saved {len(video)} frames to {image_path.split('.')[0]}.mp4")
