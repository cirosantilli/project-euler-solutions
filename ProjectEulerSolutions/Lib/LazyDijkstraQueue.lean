import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Int.Order.Basic

/-! Coverage of current distances by a priority queue that retains stale entries.
Heap order is separate: these properties depend only on queue membership. -/
namespace ProjectEulerSolutions.LazyDijkstraQueue

variable {α : Type}

structure Invariant (valid : α → Prop) (infinity : Int) (distance : α → Int)
    (visited : α → Bool) (queue : List (Int × α)) : Prop where
  valid_entry : ∀ entry ∈ queue, valid entry.2
  lower : ∀ entry ∈ queue, distance entry.2 ≤ entry.1
  finite : ∀ entry ∈ queue, entry.1 < infinity
  covered : ∀ v, valid v → visited v = false → distance v < infinity → (distance v, v) ∈ queue

theorem Invariant.perm {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {queue other : List (Int × α)}
    (invariant : Invariant valid infinity distance visited queue) (permutation : other.Perm queue) :
    Invariant valid infinity distance visited other := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro entry member
    exact invariant.valid_entry entry (permutation.mem_iff.mp member)
  · intro entry member
    exact invariant.lower entry (permutation.mem_iff.mp member)
  · intro entry member
    exact invariant.finite entry (permutation.mem_iff.mp member)
  · intro v inside waiting finite
    exact permutation.mem_iff.mpr (invariant.covered v inside waiting finite)

/-- A minimum pop for an unvisited vertex is its current key, despite stale duplicates. -/
theorem Invariant.minimum_current {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {root : Int × α} {rest : List (Int × α)}
    (invariant : Invariant valid infinity distance visited (root :: rest))
    (minimum : ∀ entry ∈ root :: rest, root.1 ≤ entry.1) (waiting : visited root.2 = false) :
    root.1 = distance root.2 ∧
      ∀ v, valid v → visited v = false → root.1 ≤ distance v := by
  have rootMember : root ∈ root :: rest := by simp
  have finite := invariant.finite root rootMember
  have lower := invariant.lower root rootMember
  have current := invariant.covered root.2 (invariant.valid_entry root rootMember) waiting (by omega)
  have upper := minimum (distance root.2, root.2) current
  refine ⟨by simpa using Int.le_antisymm upper lower, ?_⟩
  intro v inside pending
  by_cases below : distance v < infinity
  · exact minimum (distance v, v) (invariant.covered v inside pending below)
  · omega

/-- Marking the popped vertex removes the only coverage obligation its pop could break. -/
theorem Invariant.pop_mark [DecidableEq α] {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {root : Int × α} {rest : List (Int × α)}
    (invariant : Invariant valid infinity distance visited (root :: rest)) :
    Invariant valid infinity distance (Function.update visited root.2 true) rest := by
  refine ⟨fun e h => invariant.valid_entry e (by simp [h]),
    fun e h => invariant.lower e (by simp [h]), fun e h => invariant.finite e (by simp [h]), ?_⟩
  intro v inside waiting finite
  have different : v ≠ root.2 := by intro same; subst v; simp at waiting
  have oldWaiting : visited v = false := by simpa [Function.update_apply, different] using waiting
  have member := invariant.covered v inside oldWaiting finite
  rcases List.mem_cons.mp member with same | tail
  · have equal := congrArg Prod.snd same
    exact False.elim (different equal)
  · exact tail

theorem Invariant.drop_visited [DecidableEq α] {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {root : Int × α} {rest : List (Int × α)}
    (invariant : Invariant valid infinity distance visited (root :: rest)) (done : visited root.2 = true) :
    Invariant valid infinity distance visited rest := by
  have unchanged : Function.update visited root.2 true = visited := by
    funext v
    by_cases same : v = root.2 <;> simp [Function.update_apply, same, done]
  simpa only [unchanged] using invariant.pop_mark

/-- A successful relaxation retains old keys and appends the new current key. -/
theorem Invariant.improve [DecidableEq α] {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {queue : List (Int × α)}
    (invariant : Invariant valid infinity distance visited queue) (v : α) (candidate : Int)
    (inside : valid v) (smaller : candidate < distance v) (finite : candidate < infinity) :
    Invariant valid infinity (Function.update distance v candidate) visited (queue ++ [(candidate, v)]) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro entry member
    rcases List.mem_append.mp member with old | new
    · exact invariant.valid_entry entry old
    · obtain rfl := List.mem_singleton.mp new
      exact inside
  · intro entry member
    rcases List.mem_append.mp member with old | new
    · have bound := invariant.lower entry old
      by_cases same : entry.2 = v
      · simp only [Function.update_apply, same, ↓reduceIte]
        rw [same] at bound
        omega
      · simpa only [Function.update_apply, if_neg same] using bound
    · obtain rfl := List.mem_singleton.mp new
      simp
  · intro entry member
    rcases List.mem_append.mp member with old | new
    · exact invariant.finite entry old
    · obtain rfl := List.mem_singleton.mp new
      exact finite
  · intro u validU waiting below
    by_cases same : u = v
    · subst u
      simp
    · have current : Function.update distance v candidate u = distance u := by simp [same]
      rw [current] at below ⊢
      exact List.mem_append_left _ (invariant.covered u validU waiting below)

theorem initial [DecidableEq α] (valid : α → Prop) (source : α) (infinity start : Int)
    (inside : valid source) (finite : start < infinity) :
    Invariant valid infinity (fun v => if v = source then start else infinity)
      (fun _ => false) [(start, source)] := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro entry member
    obtain rfl := List.mem_singleton.mp member
    exact inside
  · intro entry member
    obtain rfl := List.mem_singleton.mp member
    simp
  · intro entry member
    obtain rfl := List.mem_singleton.mp member
    exact finite
  · intro v _ _ below
    by_cases same : v = source
    · subst v
      simp
    · simp [same] at below

theorem Invariant.nonempty {valid : α → Prop} {infinity : Int} {distance : α → Int}
    {visited : α → Bool} {queue : List (Int × α)}
    (invariant : Invariant valid infinity distance visited queue) (v : α)
    (inside : valid v) (waiting : visited v = false) (finite : distance v < infinity) : queue ≠ [] := by
  intro empty
  have member := invariant.covered v inside waiting finite
  simp [empty] at member

/-- With at most `fanout` new keys per settled vertex, this potential strictly decreases. -/
theorem fresh_potential_decreases (fanout pending nextPending remaining nextLength : Nat)
    (settled : nextPending < pending) (pushed : nextLength ≤ remaining + fanout) :
    (fanout + 1) * nextPending + nextLength < (fanout + 1) * pending + (remaining + 1) := by
  have bound := Nat.mul_le_mul_left (fanout + 1) (show nextPending + 1 ≤ pending by omega)
  rw [Nat.mul_add, Nat.mul_one] at bound
  omega

theorem stale_potential_decreases (fanout pending remaining : Nat) :
    (fanout + 1) * pending + remaining < (fanout + 1) * pending + (remaining + 1) := by omega

end ProjectEulerSolutions.LazyDijkstraQueue
