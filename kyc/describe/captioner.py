class CaptionGenerator:
    ACTION_RULES = [
        ({"person", "bicycle"}, "A person riding a bicycle."),
        ({"person", "motorcycle"}, "A person riding a motorcycle."),
        ({"person", "horse"}, "A person riding a horse."),
        ({"person", "skis"}, "A person skiing."),
        ({"person", "snowboard"}, "A person snowboarding."),
        ({"person", "surfboard"}, "A person surfing."),
        ({"person", "tennis racket"}, "A person playing tennis."),
        ({"person", "sports ball"}, "A person playing with a ball."),
        ({"dog", "sports ball"}, "A dog playing with a ball."),
        ({"cat", "couch"}, "A cat resting on a couch."),
        ({"car", "traffic light"}, "Cars near a traffic light."),
        ({"car", "truck"}, "Vehicles on or near a road."),
        ({"boat", "person"}, "People near a boat."),
    ]

    SCENE_HINTS = {
        "street": {"car", "bus", "truck", "traffic light", "stop sign", "motorcycle", "bicycle"},
        "kitchen": {"bottle", "cup", "fork", "knife", "spoon", "bowl", "oven", "sink", "refrigerator"},
        "office": {"laptop", "keyboard", "mouse", "cell phone", "book", "chair"},
        "living room": {"couch", "tv", "chair", "potted plant", "dining table"},
        "outdoor": {"bird", "dog", "horse", "sheep", "cow", "boat", "bench", "sports ball"},
    }

    def generate(self, detections, text=""):
        if not detections:
            if text:
                return f"An image containing readable text: {text}"
            return "No clear objects detected in the image."

        objects = [d["label"] for d in detections[:8]]

        # remove duplicates while preserving order
        seen = set()
        unique_objects = []
        for obj in objects:
            if obj not in seen:
                unique_objects.append(obj)
                seen.add(obj)

        object_set = set(unique_objects)
        caption = self._action_caption(object_set)

        if caption is None and len(unique_objects) == 1:
            caption = f"An image showing {self._article_safe(unique_objects[0])}."
        elif caption is None and len(unique_objects) == 2:
            caption = f"An image showing {self._article_safe(unique_objects[0])} and {self._article_safe(unique_objects[1])}."
        elif caption is None:
            caption = (
                f"An image showing {', '.join(unique_objects[:-1])}, "
                f"and {unique_objects[-1]}."
            )

        scene = self.scene_tag(unique_objects)
        if scene:
            caption = caption[:-1] + f" in {self._article_safe(scene)} scene."

        if text:
            caption += f" Readable text says: {text}"

        return caption

    def tags(self, detections, text=""):
        unique_objects = []
        seen = set()
        for detection in detections:
            label = detection["label"]
            if label not in seen:
                unique_objects.append(label)
                seen.add(label)

        scene = self.scene_tag(unique_objects)
        tags = unique_objects[:10]
        if scene and scene not in tags:
            tags.append(scene)
        if text:
            tags.append("text")
        return tags

    def scene_tag(self, objects):
        object_set = set(objects)
        best_scene = ""
        best_score = 0
        for scene, labels in self.SCENE_HINTS.items():
            score = len(object_set.intersection(labels))
            if score > best_score:
                best_score = score
                best_scene = scene
        return best_scene if best_score >= 2 else ""

    def _action_caption(self, object_set):
        for required, caption in self.ACTION_RULES:
            if required.issubset(object_set):
                return caption
        return None

    def _article_safe(self, label):
        if label.endswith("s") or " " in label:
            return label
        article = "an" if label[0].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {label}"
