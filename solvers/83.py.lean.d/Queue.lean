import solvers.«83.py.lean.d».Inputs
import ProjectEulerSolutions.Lib.BinaryHeapOrder
import lang.py.FormalSW.HeapqTuples

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def QueueEntry (heap : Mutable.Heap) (value : Mutable.Value) : Prop :=
  ∃ address : Nat, ∃ distance : Int, ∃ vertex : Nat, value = .ref address ∧
    heap[address]? = some (.tuple [.scalar (.int distance), .scalar (.int vertex)])

def priority (heap : Mutable.Heap) : Mutable.Value → Int
  | .ref address => match heap[address]? with
      | some (.tuple (.scalar (.int distance) :: _)) => distance
      | _ => 0
  | _ => 0

theorem QueueEntry.integerTuple {heap : Mutable.Heap} {value : Mutable.Value}
    (typed : QueueEntry heap value) : Heapq.IntegerTuple heap value := by
  obtain ⟨address, distance, vertex, rfl, found⟩ := typed
  exact ⟨address, [distance, vertex], rfl, found⟩

theorem priority_read (heap : Mutable.Heap) (address : Nat) (distance : Int) (vertex : Nat)
    (found : heap[address]? = some (.tuple [.scalar (.int distance), .scalar (.int vertex)])) :
    priority heap (.ref address) = distance := by simp [priority, found]

theorem QueueEntry.frame {before after : Mutable.Heap} {value : Mutable.Value}
    (typed : QueueEntry before value) (kept : Heapq.TupleFrame before after) :
    QueueEntry after value ∧ priority after value = priority before value := by
  obtain ⟨address, distance, vertex, rfl, found⟩ := typed
  have retained := kept address [distance, vertex] found
  exact ⟨⟨address, distance, vertex, rfl, retained⟩,
    (priority_read after address distance vertex retained).trans
      (priority_read before address distance vertex found).symm⟩

theorem queue_pair_comparison (heap : Mutable.Heap) (a b : Nat) (distance otherDistance : Int)
    (vertex otherVertex : Nat)
    (left : heap[a]? = some (.tuple [.scalar (.int distance), .scalar (.int vertex)]))
    (right : heap[b]? = some (.tuple [.scalar (.int otherDistance), .scalar (.int otherVertex)])) :
    Heapq.less (.ref a) (.ref b) heap =
      some (.ok (if distance = otherDistance then decide (vertex < otherVertex)
        else decide (distance < otherDistance)), heap) := by
  simpa using Heapq.less_int_pair heap a b distance vertex
    otherDistance otherVertex left right

/-- A concrete pop preserves tuple identities and returns a minimum priority from an ordered queue. -/
theorem queue_pop_result (heap : Mutable.Heap) (address : Nat) (root : Mutable.Value)
    (rest : List Mutable.Value) (env : HeapCalls.Env)
    (localHeap : env "heap" = some (.ref address))
    (found : heap[address]? = some (.list (root :: rest)))
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (ordered : ProjectEulerSolutions.BinaryHeapOrder.Ordered (priority heap) (root :: rest)) :
    ∃ after, Evaluates program env (.builtinCall .heapqPop [.name "heap"]) heap root after ∧
      Heapq.Reorders heap address rest after ∧
      Heapq.TupleFrame heap after ∧
      ∀ value ∈ root :: rest, priority heap root ≤ priority heap value := by
  obtain ⟨after, executed, contents⟩ := Heapq.heappop_int_tuples heap address root rest found
    (fun value member => (typed value member).integerTuple)
  refine ⟨after, Evaluates.builtin (.cons (.lookup localHeap) .nil) executed, contents,
    (Heapq.TupleFrame.replace_list heap address (root :: rest) found).reorders
      (Heapq.TupleFrame.refl heap) contents, ?_⟩
  exact ProjectEulerSolutions.BinaryHeapOrder.root_le_member (priority heap) root rest ordered

/-- Pushing a tuple changes only the queue list and retains duplicate references. -/
theorem queue_push_contents (heap : Mutable.Heap) (address : Nat) (values : List Mutable.Value)
    (item : Mutable.Value) (found : heap[address]? = some (.list values))
    (typed : ∀ value ∈ values, QueueEntry heap value) (itemTyped : QueueEntry heap item) :
    ∃ after, Heapq.heappush [.ref address, item] heap = some (.ok (.scalar .none), after) ∧
      Heapq.Reorders heap address (values ++ [item]) after :=
  Heapq.heappush_int_tuples heap address values item found
    (fun value member => (typed value member).integerTuple) itemTyped.integerTuple

end solvers.«83»
