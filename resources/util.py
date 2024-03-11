import pyray as rl
import glm
from typing import Optional

def copy_rect(rect):
    return rl.Rectangle(rect.x, rect.y, rect.width, rect.height)


def get_signed_collision_rec(rect1: rl.Rectangle, rect2: rl.Rectangle) -> rl.Rectangle:
    """Compute the rectangle of intersection between rect1 and rect2.

    If rect2 is to the left or above rect1, the width or height will
    be flipped, respectively."""
    r = rl.get_collision_rec(rect1, rect2)
    if rect2.x < rect1.x:
        r.width = -r.width
    if rect2.y < rect1.y:
        r.height = -r.height
    return r


def resolve_map_collision(map_aabbs, actor_aabb) -> Optional[glm.vec2]:
    """Fix overlap with map tiles. Returns new position for actor_aabb."""
    # internal copy of actor_aabb that will be mutated
    aabb = copy_rect(actor_aabb)
    if map_aabbs:
        for i in range(3):  # run multiple iters to handle corners/passages
            most_overlap = max(
                (get_signed_collision_rec(r, aabb) for r in map_aabbs),
                key=lambda x: abs(x.width * x.height),
            )
            if abs(most_overlap.width) < abs(most_overlap.height):
                aabb.x += most_overlap.width
            else:
                aabb.y += most_overlap.height

    new_pos = glm.vec2(aabb.x, aabb.y)
    old_pos = (actor_aabb.x, actor_aabb.y)
    return new_pos if new_pos != old_pos else None
