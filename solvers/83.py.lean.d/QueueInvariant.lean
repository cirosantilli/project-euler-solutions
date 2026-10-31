import solvers.«83.py.lean.d».OrderedQueue
import ProjectEulerSolutions.Lib.LazyDijkstraQueue

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.BinaryHeapOrder
set_option Elab.async false

def vertex (heap : Mutable.Heap) : Mutable.Value → Nat
  | .ref address => match heap[address]? with
      | some (.tuple [_, .scalar (.int index)]) => index.toNat
      | _ => 0
  | _ => 0

def queueKeys (heap : Mutable.Heap) (values : List Mutable.Value) : List (Int × Nat) :=
  values.map fun value => (priority heap value, vertex heap value)

theorem vertex_read (heap : Mutable.Heap) (address : Nat) (distance : Int) (index : Nat)
    (found : heap[address]? = some (.tuple [.scalar (.int distance), .scalar (.int index)])) :
    vertex heap (.ref address) = index := by simp [vertex, found]

theorem QueueEntry.vertex_frame {before after : Mutable.Heap} {value : Mutable.Value}
    (typed : QueueEntry before value) (kept : Heapq.TupleFrame before after) :
    vertex after value = vertex before value := by
  obtain ⟨address, distance, index, rfl, found⟩ := typed
  exact (vertex_read after address distance index (kept address [distance, index] found)).trans
    (vertex_read before address distance index found).symm

theorem queueKeys_frame (before after : Mutable.Heap) (values : List Mutable.Value)
    (typed : ∀ value ∈ values, QueueEntry before value) (kept : Heapq.TupleFrame before after) :
    queueKeys after values = queueKeys before values := by
  apply List.map_congr_left
  intro value member
  rw [((typed value member).frame kept).2, (typed value member).vertex_frame kept]

theorem queueKeys_reorders (heap after : Mutable.Heap) (address : Nat)
    (original values result : List Mutable.Value)
    (found : heap[address]? = some (.list original))
    (typed : ∀ value ∈ values, QueueEntry heap value)
    (reordered : Heapq.Reorders heap address values after)
    (finalFound : after[address]? = some (.list result)) :
    (queueKeys after result).Perm (queueKeys heap values) ∧
      ∀ value ∈ result, QueueEntry after value := by
  have kept := (Heapq.TupleFrame.replace_list heap address original found).reorders
    (Heapq.TupleFrame.refl heap) reordered
  obtain ⟨changed, changedFound, permutation⟩ :=
    Heapq.Reorders.contents heap address values after (List.getElem?_eq_some_iff.mp found).1 reordered
  have same : changed = result := by simpa only [finalFound, Option.some.injEq,
    Mutable.Object.list.injEq] using changedFound.symm
  subst changed
  have baseTyped : ∀ value ∈ result, QueueEntry heap value :=
    fun value member => typed value (permutation.mem_iff.mp member)
  refine ⟨?_, fun value member => ((baseTyped value member).frame kept).1⟩
  rw [queueKeys_frame heap after result baseTyped kept]
  exact permutation.map _

def CoveredQueue (heap : Mutable.Heap) (size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (values : List Mutable.Value) : Prop :=
  ProjectEulerSolutions.LazyDijkstraQueue.Invariant (fun v => v < size) infinity distance visited
    (queueKeys heap values)

/-- The old root's key is current whenever its vertex has not yet been settled. -/
theorem CoveredQueue.minimum_current (heap : Mutable.Heap) (size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (root : Mutable.Value) (rest : List Mutable.Value)
    (covered : CoveredQueue heap size infinity distance visited (root :: rest))
    (ordered : Ordered (priority heap) (root :: rest)) (waiting : visited (vertex heap root) = false) :
    priority heap root = distance (vertex heap root) ∧
      ∀ v, v < size → visited v = false → priority heap root ≤ distance v := by
  apply ProjectEulerSolutions.LazyDijkstraQueue.Invariant.minimum_current covered ?_ waiting
  intro entry member
  obtain ⟨value, present, rfl⟩ := (List.mem_map
    (f := fun value => (priority heap value, vertex heap value)) (l := root :: rest)).mp member
  exact root_le_member (priority heap) root rest ordered value present

theorem CoveredQueue.pop_mark (heap after : Mutable.Heap) (address size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (root : Mutable.Value) (rest result : List Mutable.Value)
    (found : heap[address]? = some (.list (root :: rest)))
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (covered : CoveredQueue heap size infinity distance visited (root :: rest))
    (reordered : Heapq.Reorders heap address rest after)
    (finalFound : after[address]? = some (.list result)) :
    CoveredQueue after size infinity distance (Function.update visited (vertex heap root) true) result := by
  have permutation := (queueKeys_reorders heap after address (root :: rest) rest result found
    (fun value member => typed value (List.mem_cons_of_mem root member)) reordered finalFound).1
  exact (ProjectEulerSolutions.LazyDijkstraQueue.Invariant.pop_mark covered).perm permutation

theorem CoveredQueue.drop_visited (heap after : Mutable.Heap) (address size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (root : Mutable.Value) (rest result : List Mutable.Value)
    (found : heap[address]? = some (.list (root :: rest)))
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (covered : CoveredQueue heap size infinity distance visited (root :: rest))
    (done : visited (vertex heap root) = true)
    (reordered : Heapq.Reorders heap address rest after)
    (finalFound : after[address]? = some (.list result)) :
    CoveredQueue after size infinity distance visited result := by
  have permutation := (queueKeys_reorders heap after address (root :: rest) rest result found
    (fun value member => typed value (List.mem_cons_of_mem root member)) reordered finalFound).1
  exact (ProjectEulerSolutions.LazyDijkstraQueue.Invariant.drop_visited covered done).perm permutation

theorem CoveredQueue.frame (heap after : Mutable.Heap) (size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (values : List Mutable.Value)
    (typed : ∀ value ∈ values, QueueEntry heap value) (kept : Heapq.TupleFrame heap after)
    (covered : CoveredQueue heap size infinity distance visited values) :
    CoveredQueue after size infinity distance visited values := by
  unfold CoveredQueue
  rw [queueKeys_frame heap after values typed kept]
  exact covered

/-- Pushing a better distance restores coverage of its updated vertex while retaining stale keys. -/
theorem CoveredQueue.push_improvement (heap : Mutable.Heap) (address size : Nat) (infinity : Int)
    (distance : Nat → Int) (visited : Nat → Bool) (values : List Mutable.Value) (item : Mutable.Value)
    (found : heap[address]? = some (.list values))
    (typed : ∀ value ∈ values, QueueEntry heap value) (itemTyped : QueueEntry heap item)
    (ordered : Ordered (priority heap) values)
    (covered : CoveredQueue heap size infinity distance visited values)
    (inside : vertex heap item < size) (smaller : priority heap item < distance (vertex heap item))
    (finite : priority heap item < infinity) :
    ∃ after result, Heapq.heappush [.ref address, item] heap = some (.ok (.scalar .none), after) ∧
      after[address]? = some (.list result) ∧ Ordered (priority after) result ∧
      (∀ value ∈ result, QueueEntry after value) ∧
      CoveredQueue after size infinity (Function.update distance (vertex heap item) (priority heap item)) visited result ∧
      Heapq.Reorders heap address (values ++ [item]) after := by
  obtain ⟨after, result, ran, finalFound, finalOrder, reordered⟩ :=
    queue_push_ordered heap address values item found typed itemTyped ordered
  have appended : ∀ value ∈ values ++ [item], QueueEntry heap value := by
    intro value member
    rcases List.mem_append.mp member with old | new
    · exact typed value old
    · obtain rfl := List.mem_singleton.mp new
      exact itemTyped
  have preserved := queueKeys_reorders heap after address values (values ++ [item]) result found appended reordered finalFound
  have improved := ProjectEulerSolutions.LazyDijkstraQueue.Invariant.improve covered (vertex heap item)
    (priority heap item) inside smaller finite
  have permutation : (queueKeys after result).Perm
      (queueKeys heap values ++ [(priority heap item, vertex heap item)]) := by
    simpa only [queueKeys, List.map_append, List.map_cons, List.map_nil] using preserved.1
  exact ⟨after, result, ran, finalFound, finalOrder, preserved.2, improved.perm permutation, reordered⟩

end solvers.«83»
