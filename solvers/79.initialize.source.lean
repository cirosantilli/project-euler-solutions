import solvers.«79.vertex.source»

/-! The graph builder allocates its three initially empty containers after the caller's objects. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev buildCode : List Stmt := executableBody
  ((findFunctionIn program.body "build_graph").getD ([], [])).2

def initialBody : List Stmt :=
  [.annAssign (.name "nodes") (.subscript (.name "set") (.name "str"))
      (some (.builtinCall .set [])),
   .annAssign (.name "adj")
      (.subscript (.name "dict")
        (.tuple [.name "str", .subscript (.name "set") (.name "str")])) (some (.dict [])),
   .annAssign (.name "indeg") (.subscript (.name "dict") (.tuple [.name "str", .name "int"]))
      (some (.dict []))]

theorem initial_source : buildCode.take 3 = initialBody := rfl

def initialRoots (heap : Mutable.Heap) : Roots :=
  ⟨heap.length, heap.length + 1, heap.length + 2⟩

def emptyStorage : Storage := ⟨[], [], [], [], fun _ => 0, fun _ => []⟩

def initialHeap (heap : Mutable.Heap) : Mutable.Heap := heap ++ [.set [], .dict [], .dict []]

def initialEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "nodes" (.ref heap.length))
    "adj" (.ref (heap.length + 1))) "indeg" (.ref (heap.length + 2))

theorem initial_represents (heap : Mutable.Heap) :
    Represents (initialRoots heap) emptyStorage (initialHeap heap) := by
  constructor
  · simp [initialRoots, initialHeap, emptyStorage, stringSet]
  · simp [initialRoots, initialHeap, emptyStorage]
  · simp [initialRoots, initialHeap, emptyStorage]
  · simp [initialRoots]
  · exact List.nodup_nil
  · exact List.nodup_nil
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]

theorem initial_block (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env initialBody heap (initialEnv env heap) (initialHeap heap) := by
  have nodes : BlockContinues program env [initialBody[0]] heap
      (setLocal env "nodes" (.ref heap.length)) (heap ++ [.set []]) := by
    apply BlockContinues.annAssign
    exact Evaluates.builtin .nil rfl
  have adjacency : BlockContinues program (setLocal env "nodes" (.ref heap.length))
      [initialBody[1]] (heap ++ [.set []])
      (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
      (heap ++ [.set [], .dict []]) := by
    apply BlockContinues.annAssign
    have evaluated : Evaluates program (setLocal env "nodes" (.ref heap.length)) (.dict [])
        (heap ++ [.set []]) (.ref (heap ++ [Mutable.Object.set []]).length)
        ((heap ++ [.set []]) ++ [.dict []]) := Evaluates.dictionary .nil
    simpa only [List.length_append, List.length_cons, List.length_nil, Nat.zero_add,
      List.append_assoc, List.cons_append, List.nil_append] using evaluated
  have degrees : BlockContinues program
      (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
      [initialBody[2]] (heap ++ [.set [], .dict []]) (initialEnv env heap)
      (initialHeap heap) := by
    apply BlockContinues.annAssign
    have evaluated : Evaluates program
        (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
        (.dict []) (heap ++ [.set [], .dict []])
        (.ref (heap ++ [Mutable.Object.set [], .dict []]).length)
        ((heap ++ [.set [], .dict []]) ++ [.dict []]) := Evaluates.dictionary .nil
    simpa only [initialEnv, initialHeap, List.length_append, List.length_cons, List.length_nil,
      Nat.zero_add, List.append_assoc, List.cons_append, List.nil_append] using evaluated
  exact nodes.append (adjacency.append degrees)

theorem initialize_graph (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env (buildCode.take 3) heap (initialEnv env heap) (initialHeap heap) ∧
      Represents (initialRoots heap) emptyStorage (initialHeap heap) :=
  ⟨initial_block env heap, initial_represents heap⟩

end solvers.«79».graph
