# Vision encoder spatial diagnostic

Selected complete records: 10
Checks passed: 144

The CPU audit independently loads full logical tensors from both models, recomputes temporal/spatial FFT energy, verifies static/gray temporal repeats, and checks an exact space-to-depth inverse. The repacking control preserves feature values and total energy while changing the spatial coordinate partition; it is not evidence of information loss.

- qwen / video_1024 / gray / vision_final_premerger: shape [8, 12, 20, 1024], temporal AC fraction 0, spatial AC fraction 0.354005; packed shape [8, 6, 10, 4096], packed spatial AC 0.348078, exact inverse=True.
- qwen / video_1024 / gray / vision_final_postmerger: shape [8, 6, 10, 2048], temporal AC fraction 0, spatial AC fraction 0.472683; packed shape [8, 3, 5, 8192], packed spatial AC 0.426462, exact inverse=True.
- qwen / video_1024 / real / vision_final_premerger: shape [8, 12, 20, 1024], temporal AC fraction 0.205001, spatial AC fraction 0.352624; packed shape [8, 6, 10, 4096], packed spatial AC 0.346186, exact inverse=True.
- qwen / video_1024 / real / vision_final_postmerger: shape [8, 6, 10, 2048], temporal AC fraction 0.271735, spatial AC fraction 0.583351; packed shape [8, 3, 5, 8192], packed spatial AC 0.557, exact inverse=True.
- qwen / video_1024 / static_first / vision_final_premerger: shape [8, 12, 20, 1024], temporal AC fraction 0, spatial AC fraction 0.360066; packed shape [8, 6, 10, 4096], packed spatial AC 0.355167, exact inverse=True.
- qwen / video_1024 / static_first / vision_final_postmerger: shape [8, 6, 10, 2048], temporal AC fraction 0, spatial AC fraction 0.607047; packed shape [8, 3, 5, 8192], packed spatial AC 0.583355, exact inverse=True.
- qwen / video_341 / gray / vision_final_premerger: shape [8, 12, 20, 1024], temporal AC fraction 0, spatial AC fraction 0.354005; packed shape [8, 6, 10, 4096], packed spatial AC 0.348078, exact inverse=True.
- qwen / video_341 / gray / vision_final_postmerger: shape [8, 6, 10, 2048], temporal AC fraction 0, spatial AC fraction 0.472683; packed shape [8, 3, 5, 8192], packed spatial AC 0.426462, exact inverse=True.
- qwen / video_341 / real / vision_final_premerger: shape [8, 12, 20, 1024], temporal AC fraction 0.281602, spatial AC fraction 0.351889; packed shape [8, 6, 10, 4096], packed spatial AC 0.346312, exact inverse=True.
- qwen / video_341 / real / vision_final_postmerger: shape [8, 6, 10, 2048], temporal AC fraction 0.393613, spatial AC fraction 0.560159; packed shape [8, 3, 5, 8192], packed spatial AC 0.53428, exact inverse=True.
- smol / video_1024 / gray / vision_final_postnorm: shape [16, 32, 32, 768], temporal AC fraction 0, spatial AC fraction 0.610746; packed shape [16, 8, 8, 12288], packed spatial AC 0.544989, exact inverse=True.
- smol / video_1024 / gray / vision_connector_output: shape [16, 8, 8, 960], temporal AC fraction 0, spatial AC fraction 0.318202; packed shape [16, 2, 2, 15360], packed spatial AC 0.196219, exact inverse=True.
- smol / video_1024 / real / vision_final_postnorm: shape [16, 32, 32, 768], temporal AC fraction 0.337196, spatial AC fraction 0.759444; packed shape [16, 8, 8, 12288], packed spatial AC 0.732042, exact inverse=True.
- smol / video_1024 / real / vision_connector_output: shape [16, 8, 8, 960], temporal AC fraction 0.136474, spatial AC fraction 0.464169; packed shape [16, 2, 2, 15360], packed spatial AC 0.350019, exact inverse=True.
- smol / video_1024 / static_first / vision_final_postnorm: shape [16, 32, 32, 768], temporal AC fraction 0, spatial AC fraction 0.739417; packed shape [16, 8, 8, 12288], packed spatial AC 0.713851, exact inverse=True.
- smol / video_1024 / static_first / vision_connector_output: shape [16, 8, 8, 960], temporal AC fraction 0, spatial AC fraction 0.443634; packed shape [16, 2, 2, 15360], packed spatial AC 0.329461, exact inverse=True.
- smol / video_341 / gray / vision_final_postnorm: shape [16, 32, 32, 768], temporal AC fraction 0, spatial AC fraction 0.610746; packed shape [16, 8, 8, 12288], packed spatial AC 0.544989, exact inverse=True.
- smol / video_341 / gray / vision_connector_output: shape [16, 8, 8, 960], temporal AC fraction 0, spatial AC fraction 0.318202; packed shape [16, 2, 2, 15360], packed spatial AC 0.196219, exact inverse=True.
- smol / video_341 / real / vision_final_postnorm: shape [16, 32, 32, 768], temporal AC fraction 0.504, spatial AC fraction 0.772226; packed shape [16, 8, 8, 12288], packed spatial AC 0.746437, exact inverse=True.
- smol / video_341 / real / vision_connector_output: shape [16, 8, 8, 960], temporal AC fraction 0.255684, spatial AC fraction 0.477109; packed shape [16, 2, 2, 15360], packed spatial AC 0.368737, exact inverse=True.
