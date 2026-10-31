import solvers.«83.py.lean.d».Queue
import ProjectEulerSolutions.Lib.BinaryHeapSift

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions.BinaryHeapOrder
set_option Elab.async false
attribute [local simp] Except.map

theorem queue_comparison_order (base heap : Mutable.Heap) (left right : Mutable.Value)
    (kept : Heapq.TupleFrame base heap) (leftTyped : QueueEntry base left)
    (rightTyped : QueueEntry base right) :
    ∃ answer, Heapq.less left right heap = some (.ok answer, heap) ∧
      (answer = true → priority base left ≤ priority base right) ∧
      (answer = false → priority base right ≤ priority base left) := by
  obtain ⟨a, distance, vertex, rfl, leftFound⟩ := leftTyped
  obtain ⟨b, otherDistance, otherVertex, rfl, rightFound⟩ := rightTyped
  refine ⟨_, queue_pair_comparison heap a b distance otherDistance vertex otherVertex
    (kept a [distance, vertex] leftFound) (kept b [otherDistance, otherVertex] rightFound), ?_, ?_⟩
  all_goals
    rw [priority_read base a distance vertex leftFound,
      priority_read base b otherDistance otherVertex rightFound]
    split_ifs with same
    · omega
    · simp only [decide_eq_true_eq, decide_eq_false_iff_not]
      omega

/-- The upward phase executes on the actual tuple queue and repairs every parent edge. -/
theorem queue_siftDown_ordered (base heap : Mutable.Heap) (address position : Nat)
    (values : List Mutable.Value)
    (stable : Heapq.ListReplacementInvariant address (Heapq.TupleFrame base))
    (kept : Heapq.TupleFrame base heap)
    (typed : ∀ value ∈ values, QueueEntry base value)
    (found : heap[address]? = some (.list values)) (inside : position < values.length)
    (ordered : UpHole values.length position (listWeight (priority base) values)) :
    ∃ after result, Heapq.siftDown address 0 position heap = some (.ok (), after) ∧
      after[address]? = some (.list result) ∧ Ordered (priority base) result := by
  by_cases moving : 0 < position
  · have smaller := parent_lt moving
    have parentInside : parent position < values.length := by omega
    obtain ⟨answer, compared, yes, no⟩ := queue_comparison_order base heap
      values[position] values[parent position] kept
      (typed _ (List.getElem_mem inside)) (typed _ (List.getElem_mem parentInside))
    dsimp only [parent] at compared
    cases answer with
    | false =>
        refine ⟨heap, values, ?_, found, (edges_iff_ordered _ _).mp (ordered.close ?_)⟩
        · rw [Heapq.siftDown]
          simp [moving, Heapq.atIndex_list heap address position values found inside,
            Heapq.atIndex_list heap address ((position - 1) / 2) values found parentInside,
            compared]
        · intro _
          simpa only [listWeight_get _ _ _ inside, listWeight_get _ _ _ parentInside] using no rfl
    | true =>
        let changed := (values.set position values[parent position]).set (parent position) values[position]
        have permutation : changed.Perm values :=
          Heapq.swap_contents_perm values position (parent position) inside parentInside
        have valid := (List.getElem?_eq_some_iff.mp found).1
        have changedFound : (heap.set address (.list changed))[address]? = some (.list changed) := by
          simp [valid]
        have next : UpHole changed.length (parent position) (listWeight (priority base) changed) := by
          rw [permutation.length_eq, listWeight_exchange]
          exact ordered.ascend inside moving (by
            simpa only [listWeight_get _ _ _ inside, listWeight_get _ _ _ parentInside] using yes rfl)
        obtain ⟨after, result, finished, finalFound, finalOrder⟩ := queue_siftDown_ordered base
          (heap.set address (.list changed)) address (parent position) changed stable
          (stable heap changed kept) (fun value member => typed value (permutation.mem_iff.mp member))
          changedFound (by rw [permutation.length_eq]; exact parentInside) next
        dsimp only [parent, changed] at finished
        refine ⟨after, result, ?_, finalFound, finalOrder⟩
        rw [Heapq.siftDown]
        simp [moving, Heapq.atIndex_list heap address position values found inside,
          Heapq.atIndex_list heap address ((position - 1) / 2) values found parentInside,
          compared, Heapq.swap_list heap address position ((position - 1) / 2) values found inside parentInside,
          finished]
  · refine ⟨heap, values, Heapq.siftDown_stop heap address 0 position (by omega), found,
      (edges_iff_ordered _ _).mp (ordered.close ?_)⟩
    omega
termination_by position
decreasing_by exact smaller

/-- CPython's descent may move past the displaced entry's final position;
the upward phase repairs it after the descent reaches a leaf. -/
theorem queue_siftChildren_ordered (base heap : Mutable.Heap) (address size position : Nat)
    (values : List Mutable.Value)
    (stable : Heapq.ListReplacementInvariant address (Heapq.TupleFrame base))
    (kept : Heapq.TupleFrame base heap)
    (typed : ∀ value ∈ values, QueueEntry base value)
    (found : heap[address]? = some (.list values)) (sized : size = values.length)
    (inside : position < size) (ordered : Hole size position (listWeight (priority base) values)) :
    ∃ after result, Heapq.siftChildren address size 0 position heap = some (.ok (), after) ∧
      after[address]? = some (.list result) ∧ Ordered (priority base) result := by
  have descend (child : Nat) (childInside : child < size) (positive : 0 < child)
      (childOf : parent child = position)
      (least : ∀ i, i < size → 0 < i → parent i = position →
        listWeight (priority base) values child ≤ listWeight (priority base) values i) :
      ∃ after result, (do
        Heapq.swap address position child
        Heapq.siftChildren address size 0 child) heap = some (.ok (), after) ∧
        after[address]? = some (.list result) ∧ Ordered (priority base) result := by
    have smaller : position < child := childOf ▸ parent_lt positive
    have positionValid : position < values.length := by omega
    have childValid : child < values.length := by omega
    let changed := (values.set position values[child]).set child values[position]
    have permutation : changed.Perm values :=
      Heapq.swap_contents_perm values position child positionValid childValid
    have valid := (List.getElem?_eq_some_iff.mp found).1
    have changedFound : (heap.set address (.list changed))[address]? = some (.list changed) := by
      simp [valid]
    have next : Hole size child (listWeight (priority base) changed) := by
      rw [listWeight_exchange]
      exact ordered.descend childInside positive childOf least
    obtain ⟨after, result, finished, finalFound, finalOrder⟩ := queue_siftChildren_ordered base
      (heap.set address (.list changed)) address size child changed stable (stable heap changed kept)
      (fun value member => typed value (permutation.mem_iff.mp member)) changedFound
      (by rw [permutation.length_eq]; exact sized) childInside next
    exact ⟨after, result, by
      simp [Heapq.swap_list heap address position child values found positionValid childValid,
        changed, finished], finalFound, finalOrder⟩
  by_cases leftInside : 2 * position + 1 < size
  · have leftValid : 2 * position + 1 < values.length := by omega
    by_cases rightInside : 2 * position + 1 + 1 < size
    · have rightValid : 2 * position + 1 + 1 < values.length := by omega
      obtain ⟨answer, compared, yes, no⟩ := queue_comparison_order base heap
        values[2 * position + 1] values[2 * position + 1 + 1] kept
        (typed _ (List.getElem_mem leftValid)) (typed _ (List.getElem_mem rightValid))
      have candidates (i : Nat) (positive : 0 < i) (childOf : parent i = position) :
          i = 2 * position + 1 ∨ i = 2 * position + 1 + 1 := by unfold parent at childOf; omega
      cases answer with
      | false =>
          have least : ∀ i, i < size → 0 < i → parent i = position →
              listWeight (priority base) values (2 * position + 1 + 1) ≤
                listWeight (priority base) values i := by
            intro i _ positive childOf
            rcases candidates i positive childOf with rfl | rfl
            · simpa only [listWeight_get _ _ _ leftValid, listWeight_get _ _ _ rightValid] using no rfl
            · exact Int.le_refl _
          obtain ⟨after, result, finished, finalFound, finalOrder⟩ := descend (2 * position + 1 + 1)
            rightInside (by omega) (by unfold parent; omega) least
          refine ⟨after, result, ?_, finalFound, finalOrder⟩
          rw [Heapq.siftChildren]
          simpa [leftInside, rightInside,
            Heapq.atIndex_list heap address (2 * position + 1) values found leftValid,
            Heapq.atIndex_list heap address (2 * position + 1 + 1) values found rightValid,
            compared] using finished
      | true =>
          have least : ∀ i, i < size → 0 < i → parent i = position →
              listWeight (priority base) values (2 * position + 1) ≤
                listWeight (priority base) values i := by
            intro i _ positive childOf
            rcases candidates i positive childOf with rfl | rfl
            · exact Int.le_refl _
            · simpa only [listWeight_get _ _ _ leftValid, listWeight_get _ _ _ rightValid] using yes rfl
          obtain ⟨after, result, finished, finalFound, finalOrder⟩ := descend (2 * position + 1)
            leftInside (by omega) (by unfold parent; omega) least
          refine ⟨after, result, ?_, finalFound, finalOrder⟩
          rw [Heapq.siftChildren]
          simpa [leftInside, rightInside,
            Heapq.atIndex_list heap address (2 * position + 1) values found leftValid,
            Heapq.atIndex_list heap address (2 * position + 1 + 1) values found rightValid,
            compared] using finished
    · have least : ∀ i, i < size → 0 < i → parent i = position →
          listWeight (priority base) values (2 * position + 1) ≤ listWeight (priority base) values i := by
        intro i valid positive childOf
        have same : i = 2 * position + 1 := by unfold parent at childOf; omega
        subst i
        exact Int.le_refl _
      obtain ⟨after, result, finished, finalFound, finalOrder⟩ := descend (2 * position + 1)
        leftInside (by omega) (by unfold parent; omega) least
      refine ⟨after, result, ?_, finalFound, finalOrder⟩
      rw [Heapq.siftChildren]
      simpa [leftInside, rightInside] using finished
  · have leaf : UpHole values.length position (listWeight (priority base) values) := by
      rw [← sized]
      apply ordered.leaf
      intro i valid positive childOf
      unfold parent at childOf
      omega
    obtain ⟨after, result, finished, finalFound, finalOrder⟩ := queue_siftDown_ordered base heap
      address position values stable kept typed found (by omega) leaf
    refine ⟨after, result, ?_, finalFound, finalOrder⟩
    rw [Heapq.siftChildren]
    simpa [leftInside] using finished
termination_by size - position
decreasing_by omega

end solvers.«83»
