import solvers.«79.ready.source»
import ProjectEulerStatements.Lib.HeapqHeapify
import lang.py.FormalSW.DictionaryConstructorCalls

/-! Source initialization orders the ready queue, creates the output list, and copies indegrees. -/
namespace solvers.«79».ready
open FormalSW.PyAst HeapCalls
open solvers.«79».graph
open ProjectEulerStatements.MinHeap
set_option Elab.async false

def prepareBody : List Stmt :=
  [.expr (.builtinCall .heapqHeapify [.name "heap"]),
   .annAssign (.name "out") (.subscript (.name "list") (.name "str")) (some (.list [])),
   .assign [.name "indeg2"] (.builtinCall .dict [.name "indeg"])]

theorem prepare_source : (topoCode.drop 1).take 3 = prepareBody := rfl

def preparedEnv (env : HeapCalls.Env) (base : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "out" (.ref (base + 1))) "indeg2" (.ref (base + 2))

def preparedHeap (heap : Mutable.Heap) (state : Storage) (queue : List String) : Mutable.Heap :=
  heap ++ [.list (values queue), .list [], .dict state.degrees]

theorem prepare_queue (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (queue : List String) (env : HeapCalls.Env)
    (getQueue : env "heap" = some (.ref heap.length))
    (getDegrees : env "indeg" = some (.ref roots.degrees)) :
    ∃ ordered, BlockContinues program env prepareBody (heap ++ [.list (values queue)])
        (preparedEnv env heap.length) (preparedHeap heap state ordered) ∧
      ordered.Perm queue ∧ Ordered ordered := by
  obtain ⟨ordered, operation, contents, orderedHeap⟩ := heapify_ordered text strings_encodeOrder
    (heap ++ [.list (values queue)]) heap.length queue (by simp [values])
  have heapified : Heapq.heapify [.ref heap.length] (heap ++ [.list (values queue)]) =
      some (.ok (.scalar .none), heap ++ [.list (values ordered)]) := by
    simpa using operation
  have first : BlockContinues program env [prepareBody[0]]
      (heap ++ [.list (values queue)]) env (heap ++ [.list (values ordered)]) := by
    apply BlockContinues.discard
    apply Evaluates.builtin (.cons (Evaluates.lookup getQueue) .nil)
    rw [primitive_heapq_heapify]
    exact heapified
  let outputEnv := setLocal env "out" (.ref (heap.length + 1))
  have second : BlockContinues program env [prepareBody[1]]
      (heap ++ [.list (values ordered)]) outputEnv
      (heap ++ [.list (values ordered), .list []]) := by
    apply BlockContinues.annAssign
    have allocate : Evaluates program env (.list []) (heap ++ [.list (values ordered)])
        (.ref (heap ++ [Mutable.Object.list (values ordered)]).length)
        ((heap ++ [.list (values ordered)]) ++ [.list []]) := Evaluates.list .nil
    simpa only [List.length_append, List.length_cons, List.length_nil, Nat.zero_add,
      List.append_assoc, List.cons_append, List.nil_append] using allocate
  have degreeInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
  have degreeFound : (heap ++ [.list (values ordered), .list []])[roots.degrees]? =
      some (.dict state.degrees) := by
    rw [List.getElem?_append_left degreeInside]
    exact represented.degreesFound
  have third : BlockContinues program outputEnv [prepareBody[2]]
      (heap ++ [.list (values ordered), .list []]) (preparedEnv env heap.length)
      (preparedHeap heap state ordered) := by
    apply BlockContinues.assign
    have copy : Evaluates program outputEnv (.builtinCall .dict [.name "indeg"])
        (heap ++ [.list (values ordered), .list []])
        (.ref (heap ++ [Mutable.Object.list (values ordered), .list []]).length)
        ((heap ++ [.list (values ordered), .list []]) ++ [.dict state.degrees]) := by
      apply Evaluates.builtin
        (.cons (Evaluates.lookup (by simpa [outputEnv, setLocal] using getDegrees)) .nil)
      exact primitive_dict_copy _ roots.degrees state.degrees degreeFound
    simpa only [preparedHeap, List.length_append, List.length_cons, List.length_nil,
      Nat.zero_add, List.append_assoc, List.cons_append, List.nil_append] using copy
  exact ⟨ordered, first.append (second.append third), contents, orderedHeap⟩

/-- The first four source statements establish an ordered queue and separate mutable workspaces. -/
theorem initialize_topo (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ queue finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.block (topoCode.take 4 ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following
            (preparedEnv (setLocal env "heap" (.ref heap.length)) heap.length),
          stack, preparedHeap heap state queue⟩, finalUsed⟩ ∧
      queue.Perm (state.vertices.filter (eligible state)) ∧ Ordered queue := by
  obtain ⟨collected, finalUsed, collect, contents⟩ :=
    initial_result choices roots state heap represented env getNodes getDegrees
      (prepareBody ++ following) anchors stack used
  obtain ⟨queue, prepare, permutation, ordered⟩ := prepare_queue roots state heap represented
    collected (setLocal env "heap" (.ref heap.length)) (by simp [setLocal])
    (by simpa [setLocal] using getDegrees)
  refine ⟨queue, finalUsed, ?_, permutation.trans contents, ordered⟩
  have same : topoCode.take 4 = topoCode.take 1 ++ prepareBody := rfl
  rw [same, List.append_assoc]
  exact collect.trans (SetCalls.Reaches.of_heap (prepare following anchors stack) choices finalUsed)

end solvers.«79».ready
