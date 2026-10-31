import solvers.«83.py.lean.d».HeapOrder

namespace solvers.«83»
open FormalSW.PyAst
open ProjectEulerSolutions.BinaryHeapOrder
set_option Elab.async false
attribute [local simp] Except.map

theorem queue_order_frame (base after : Mutable.Heap) (values : List Mutable.Value)
    (kept : Heapq.TupleFrame base after) (typed : ∀ value ∈ values, QueueEntry base value)
    (ordered : Ordered (priority base) values) : Ordered (priority after) values := by
  intro i inside positive
  have parentInside : (i - 1) / 2 < values.length := by
    have := parent_lt positive
    unfold parent at this
    omega
  rw [((typed _ (List.getElem_mem parentInside)).frame kept).2,
    ((typed _ (List.getElem_mem inside)).frame kept).2]
  exact ordered i inside positive

theorem queue_order_reorders (heap after : Mutable.Heap) (address : Nat)
    (original values result : List Mutable.Value)
    (found : heap[address]? = some (.list original))
    (typed : ∀ value ∈ values, QueueEntry heap value)
    (reordered : Heapq.Reorders heap address values after)
    (finalFound : after[address]? = some (.list result))
    (ordered : Ordered (priority heap) result) : Ordered (priority after) result := by
  have kept := (Heapq.TupleFrame.replace_list heap address original found).reorders
    (Heapq.TupleFrame.refl heap) reordered
  obtain ⟨changed, changedFound, permutation⟩ :=
    Heapq.Reorders.contents heap address values after (List.getElem?_eq_some_iff.mp found).1 reordered
  have same : changed = result := by simpa only [finalFound, Option.some.injEq,
    Mutable.Object.list.injEq] using changedFound.symm
  subst changed
  exact queue_order_frame heap after result kept
    (fun value member => typed value (permutation.mem_iff.mp member)) ordered

/-- Push retains heap order for arbitrary integer priorities, including ties and aliases. -/
theorem queue_push_ordered (heap : Mutable.Heap) (address : Nat) (values : List Mutable.Value)
    (item : Mutable.Value) (found : heap[address]? = some (.list values))
    (typed : ∀ value ∈ values, QueueEntry heap value) (itemTyped : QueueEntry heap item)
    (ordered : Ordered (priority heap) values) :
    ∃ after result, Heapq.heappush [.ref address, item] heap = some (.ok (.scalar .none), after) ∧
      after[address]? = some (.list result) ∧ Ordered (priority after) result ∧
      Heapq.Reorders heap address (values ++ [item]) after := by
  have valid := (List.getElem?_eq_some_iff.mp found).1
  have stable := Heapq.TupleFrame.replace_list heap address values found
  have appended : ∀ value ∈ values ++ [item], QueueEntry heap value := by
    intro value member
    rcases List.mem_append.mp member with old | new
    · exact typed value old
    · simpa using (List.mem_singleton.mp new) ▸ itemTyped
  obtain ⟨after, result, finished, finalFound, finalOrder⟩ := queue_siftDown_ordered heap
    (heap.set address (.list (values ++ [item]))) address values.length (values ++ [item])
    stable (stable heap _ (Heapq.TupleFrame.refl heap)) appended (by simp [valid]) (by simp)
    (ordered_append_hole (priority heap) values item ordered)
  have ran := Heapq.heappush_result heap after address values item (.ok ()) found finished
  have reordered := Heapq.heappush_reorders heap after address values item (.ok (.scalar .none)) found ran
  exact ⟨after, result, ran, finalFound,
    queue_order_reorders heap after address values (values ++ [item]) result found appended
      reordered finalFound finalOrder, reordered⟩

/-- Pop removes the old minimum and restores order on the remaining concrete queue. -/
theorem queue_pop_ordered (heap : Mutable.Heap) (address : Nat) (root : Mutable.Value)
    (rest : List Mutable.Value) (found : heap[address]? = some (.list (root :: rest)))
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (ordered : Ordered (priority heap) (root :: rest)) :
    ∃ after result, Heapq.heappop [.ref address] heap = some (.ok root, after) ∧
      after[address]? = some (.list result) ∧ Ordered (priority after) result ∧
      Heapq.Reorders heap address rest after ∧
      ∀ value ∈ root :: rest, priority heap root ≤ priority heap value := by
  have minimum := root_le_member (priority heap) root rest ordered
  have valid := (List.getElem?_eq_some_iff.mp found).1
  rcases List.eq_nil_or_concat rest with empty | ⟨middle, last, equation⟩
  · subst rest
    refine ⟨heap.set address (.list []), [], Heapq.heappop_singleton heap address root found,
      by simp [valid], ordered_nil _, ⟨[], .refl _, rfl⟩, minimum⟩
  · have tail : rest = middle ++ [last] := by simpa only [List.concat_eq_append] using equation
    clear equation
    subst rest
    have stable := Heapq.TupleFrame.replace_list heap address (root :: (middle ++ [last])) found
    have permutation : (last :: middle).Perm (middle ++ [last]) :=
      List.perm_append_comm (l₁ := [last]) (l₂ := middle)
    have keptPrefix : Ordered (priority heap) (root :: middle) :=
      ordered_prefix (priority heap) (root :: middle) [last] ordered
    have hole : Hole (last :: middle).length 0 (listWeight (priority heap) (last :: middle)) := by
      simpa using ordered_replace_root_hole (priority heap) (root :: middle) last keptPrefix
    have changedFound : (heap.set address (.list (last :: middle)))[address]? =
        some (.list (last :: middle)) := by simp [valid]
    obtain ⟨after, result, finished, finalFound, finalOrder⟩ := queue_siftChildren_ordered heap
      (heap.set address (.list (last :: middle))) address (last :: middle).length 0 (last :: middle)
      stable (stable heap _ (Heapq.TupleFrame.refl heap))
      (fun value member => typed value (List.mem_cons_of_mem root (permutation.mem_iff.mp member)))
      changedFound rfl (by simp) hole
    have sifted : Heapq.siftUp address 0 (heap.set address (.list (last :: middle))) =
        some (.ok (), after) := by
      simpa [Heapq.siftUp, Heapq.items_list _ address (last :: middle) changedFound] using finished
    have ran : Heapq.heappop [.ref address] heap = some (.ok root, after) := by
      rw [Heapq.heappop_nontrivial heap address root last middle found]
      simp [sifted]
    have reordered := Heapq.heappop_reorders heap after address root (middle ++ [last]) (.ok root) found ran
    exact ⟨after, result, ran, finalFound,
      queue_order_reorders heap after address (root :: (middle ++ [last])) (middle ++ [last]) result
        found (fun value member => typed value (List.mem_cons_of_mem root member)) reordered finalFound finalOrder,
      reordered, minimum⟩

end solvers.«83»
