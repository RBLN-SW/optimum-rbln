# SAM3 Model Family Reference

**Transformers 5** (`transformers>=5.0.0`) — [Source](https://github.com/huggingface/transformers/tree/v5.0.0/src/transformers/models)

---

## 1. Task Comparison

| Aspect | Sam3 | Sam3Tracker | Sam3TrackerVideo | Sam3Video |
|--------|------|--------------|------------------|-----------|
| Input | Image | Image | Video | Video |
| Prompt | Text, boxes | Points, boxes, masks | Points, boxes, masks | Text |
| Task | Detect & segment by concept | Segment at location | Track across frames | Detect by text, then track |

---

## 2. Module Composition

| Module | Sam3 | Sam3Tracker | Sam3TrackerVideo | Sam3Video |
|--------|:----:|:-----------:|:----------------:|:---------:|
| vision_encoder | ✅ | ✅ | ✅* | ✅ |
| shared_image_embedding | | ✅ | ✅ | |
| text_encoder | ✅ | | | ✅ |
| text_projection | ✅ | | | ✅ |
| prompt_encoder | | ✅ | ✅ | ✅ |
| geometry_encoder | ✅ | | | ✅ |
| detr_encoder | ✅ | | | ✅ |
| detr_decoder | ✅ | | | ✅ |
| mask_decoder | ✅ | ✅ | ✅ | ✅ |
| dot_product_scoring | ✅ | | | ✅ |
| memory_attention | | | ✅ | ✅ |
| memory_encoder | | | ✅ | ✅ |
| memory_temporal_positional_encoding | | | ✅ | ✅ |
| no_memory_embedding | | ✅ | ✅ | ✅ |
| no_memory_positional_encoding | | | ✅ | ✅ |
| object_pointer_proj | | | ✅ | ✅ |
| mask_downsample | | | ✅ | ✅ |
| no_object_pointer | | | ✅ | ✅ |
| occlusion_spatial_embedding_parameter | | | ✅* | ✅ |
| temporal_positional_encoding_projection_layer | | | ✅ | ✅ |
| tracker_neck | | | | ✅ |

\* Sam3TrackerVideo: vision_encoder omitted when used in Sam3Video (shares detector). occlusion_spatial_embedding_parameter: optional.

---

## 3. Structure

```
Sam3
├── vision_encoder (backbone + neck)
├── text_encoder (CLIP)
├── geometry_encoder
├── detr_encoder / detr_decoder
├── mask_decoder
└── dot_product_scoring

Sam3Tracker
├── shared_image_embedding
├── vision_encoder
├── prompt_encoder
├── mask_decoder
└── no_memory_embedding

Sam3TrackerVideo
├── shared_image_embedding
├── vision_encoder*
├── prompt_encoder
├── mask_decoder
├── memory_attention
├── memory_encoder
├── no_memory_embedding
├── no_memory_positional_encoding
├── memory_temporal_positional_encoding
├── object_pointer_proj
├── mask_downsample
├── no_object_pointer
├── temporal_positional_encoding_projection_layer
└── occlusion_spatial_embedding_parameter*

Sam3Video
├── detector_model (Sam3)           — text-based detection
├── tracker_model (Sam3TrackerVideo) — propagation (no vision_encoder)
└── tracker_neck                    — detector vision → tracker format
```

\* vision_encoder: omitted when used in Sam3Video. occlusion_spatial_embedding_parameter: optional.
