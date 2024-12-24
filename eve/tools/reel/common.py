from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

"""
Reel spec / storyboard
"""

class Vocal(BaseModel):
    speaker: str = Field(
        ...,
        description='The name of the character speaking the line. The name of the speaker should match the name of a character in the Reel, if one exists, or "narrator" if it is a voiceover.'
    )
    text: str = Field(
        ...,
        description="What the speaker says."
    )

class VideoClip(BaseModel):
    """
    A unit of a video clip within the reel. 
    """
    scene_description: str = Field(
        ...,
        description="A purely visual description of what's happening. Focus on settings, contents, visual elements, objects, characters, and visible actions."
    )
    camera_motion: Optional[str] = Field(
        None,
        description="Brief (3-10 words) description of any specific camera motion, e.g. slow pan upward, zooming in, etc. Focus on movement, not on angles, quality, or visual elements."
    )
    sound_effects: Optional[str] = Field(
        None,
        description="Any sound effects or foley effects (e.g., 'waves crashing', 'footsteps on gravel')."
    )
    vocals: Optional[Vocal] = Field(
        None,
        description="Optional spoken dialogue from a single character or narrator. If using scene vocals, it should be a short snippet, like in scene-level dialogue between multiple characters."
    )
    duration_in_seconds: Optional[int] = Field(
        None,
        description="Estimated or desired target length of this clip in seconds. This should be between 5 and 15 seconds, and usually 10. If there are vocals, and you want the scene to be as long as the vocals, leave this blank. The only reason to set this explciitly when vocals are already set is if you want the scene to be signficiantly longer than the vocals, like for example if the vocals are just retorts or sound effects or onomatopoeia."
    )

class Character(BaseModel):
    """
    A character in the film. Useful for named characters, not NPCs.
    """
    name: str = Field(
        ...,
        description="The name of the character."
    )
    description: str = Field(
        ...,
        description="A short description of the character, and their role in the story, no more than 1-2 sentences. Avoid backstory."
    )
    visual_description: str = Field(
        None,
        description="A concise phrase of the character's look, starting with their name and describing their visual appearance, including but not limited to their type or species if not human, clothing, hair, facial features, or other distinguishing features. This should be no more than 1 sentence, 10-15 words, solely visual terms, no predicate or verbs. For example 'David, a tallman with thick rimmed glasses and a green sweater', or 'Mycos, an ethereal ceramic turquoise figurine with wings and a crown of thorns'."
    )

class ReelStoryboard(BaseModel):
    """
    A short narrative video of anywhere between 30 seconds and 2 minutes, composed of one or more Scenes.
    """
    title: Optional[str] = Field(
        None,
        description="A short and catchy name."
    )
    brief: Optional[str] = Field(
        None,
        description="A concise statement describing the reel's purpose, theme, logline, or summary."
    )
    characters: List[Character] = Field(
        ...,
        description="A list of characters in the film. Only include named characters, not NPCs or extras."
    )
    clips: List[VideoClip] = Field(
        ...,
        description="An ordered list of VideoClips that sequence the reel. A short clip can be just one of these. A more elaborate or longer video can be a combination of multiple of these which goes to 1-2 minutes, and can reach up to 5 minutes with enough instruction."
    )
    overall_voiceover: Optional[Vocal] = Field(
        None,
        description="Optional voiceover that apply across the entire set of clips. Generally best for voiceover or narration. Use with caution if also using scene-level vocals. Overall vocals dominate and are mixed in louder, so scene level vocals should be very short and used sparingly. This needs to span the entire video, so it should be roughly 20-30 words per 10 seconds of video."
    )
    music_prompt: Optional[str] = Field(
        None,
        description='A prompt describing music for the entire reel. Usually describing format, genre, sub-genre, instruments, moods, BPM, and styles, separated by |. Include specific details by combining musical and emotional terms for moods, using descriptive adjectives for instruments, and selecting BPM settings appropriate to the genre. Follow the provided examples to ensure clarity and comprehensiveness, ensuring each prompt clearly defines the desired audio output. Examples: "Format: Orchestra | Genre: Epic | Sub-genre: Cinematic Trailer | Instruments: Strings, Brass, Percussion, Choir | Moods: Dramatic, Inspiring, Heroic | Styles: Hollywood Blockbuster | BPM: 90", "Format: Electronic | Genre: Synthwave | Sub-genre: Retro-Futuristic | Instruments: Analog Synths, Drum Machine, Bass | Moods: Nostalgic, Cool, Rhythmic | Styles: 1980s Sci-Fi | BPM: 115"'
    )
    visual_prompt: str = Field(
        ...,
        description="A prompt describing the reel's overall visual style or aesthetics. This should be a concise combination of phrases describing genre, medium, texture, color palette, and style. It will apply across the entire reel. It should be from 5 to 20 words."
    )
    target_length_in_seconds: Optional[int] = Field(
        None,
        description="Estimated target length for the entire reel in seconds."
    )
    aspect_ratio: Optional[Literal["16:9", "9:16"]] = Field(
        None,
        description="The aspect ratio of the reel. Either 'landscape' or 'portrait'.",
        tip="Once this is set, you can't change it withour redoing all the video clips in the new aspect ratio."
    )


"""
Rendered reel components
"""

class MediaAsset(BaseModel):
    """
    An audio track.
    """
    url: str = Field(
        ...,
        description="The URL of the media asset."
    )

class DraftVideoClip(BaseModel):
    """
    A rendered scene clip.
    """
    image: Optional[MediaAsset] = Field(
        ...,
        description="The preview image (pre image-to-video) for the scene clip."
    )
    video: Optional[MediaAsset] = Field(
        ...,
        description="The video rendered from the image for the scene clip."
    )
    vocals: Optional[MediaAsset] = Field(
        ...,
        description="The rendered audio vocals for the scene clip."
    )
    sound_effects: Optional[MediaAsset] = Field(
        ...,
        description="The rendered sound effects for the scene clip."
    )

class DraftReel(BaseModel):
    """
    A rendered reel in preview mode.
    """
    scene_clips: List[Union[None, DraftVideoClip]] = Field(
        ...,
        description="An array of rendered or currently-unrendered scenes that are currently in the reel. These must correspond 1-to-1 with the scenes in the reel draft."
    )
    music: Optional[MediaAsset] = Field(
        ...,
        description="The rendered music for the reel."
    )
    overall_voiceover: Optional[MediaAsset] = Field(
        ...,
        description="The rendered overall voiceover for the reel."
    )

