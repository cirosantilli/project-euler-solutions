import solvers.«79.state.source»
import lang.py.FormalSW.SetComprehensionCalls
import Mathlib.Data.List.InsertIdx

/-! The initial priority queue contains exactly the zero-indegree vertices, in any set order. -/
namespace solvers.«79».ready
open FormalSW.PyAst HeapCalls
open solvers.«79».graph ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def condition : Expr :=
  .compare (.subscript (.name "indeg") (.name "v")) [(.eq, .constant (.int 0))]

def expression : Expr := .listComp (.name "v") [("v", .name "nodes", [condition])]

def body (collector : Nat) : List Stmt :=
  [.ifThen condition [.comprehensionAppend collector (.name "v")] []]

abbrev topoCode : List Stmt := executableBody
  ((findFunctionIn program.body "topo_sort").getD ([], [])).2

theorem source : topoCode[0]? = some
    (.annAssign (.name "heap") (.subscript (.name "list") (.name "str")) (some expression)) := rfl

abbrev values (items : List String) : List Mutable.Value := items.map text
def eligible (state : Storage) (vertex : String) : Bool := decide (indegree vertex state.edges = 0)
def emit (state : Storage) (collected : List String) (vertex : String) : List String :=
  if eligible state vertex then collected ++ [vertex] else collected

theorem collect_one (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (vertex : String) (member : vertex ∈ state.vertices) (collected : List String) :
    BlockContinues program (setLocal env "v" (text vertex)) (body heap.length)
      (heap ++ [.list (values collected)]) (setLocal env "v" (text vertex))
      (heap ++ [.list (values (emit state collected vertex))]) := by
  have degreesInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
  have degreesFound : (heap ++ [.list (values collected)])[roots.degrees]? =
      some (.dict state.degrees) := by
    rw [List.getElem?_append_left degreesInside]
    exact represented.degreesFound
  have read : Evaluates program (setLocal env "v" (text vertex))
      (.subscript (.name "indeg") (.name "v")) (heap ++ [.list (values collected)])
      (number (indegree vertex state.edges)) (heap ++ [.list (values collected)]) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup (by simpa [setLocal] using getDegrees))
      (Evaluates.lookup (value := text vertex) (by simp [setLocal]))
    exact Mutable.getItem_dictionary _ roots.degrees state.degrees degreesFound
      (.string vertex) (represented.counts vertex member)
  have tested : Evaluates program (setLocal env "v" (text vertex)) condition
      (heap ++ [.list (values collected)]) (.scalar (.bool (eligible state vertex)))
      (heap ++ [.list (values collected)]) := by
    apply Evaluates.compare read Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
  by_cases ready : eligible state vertex = true
  · simp only [emit, if_pos ready]
    apply BlockContinues.branch (flag := true) (by simpa only [ready] using tested) rfl
    have appended := BlockContinues.comprehension_append (module := program) (globals := [])
      (env := setLocal env "v" (text vertex)) (address := heap.length)
      (value := text vertex) (items := values collected)
      (heap := heap ++ [.list (values collected)])
      (Evaluates.lookup (name := "v") (by simp [setLocal])) (by simp)
    simpa [values] using appended
  · have notReady : eligible state vertex = false := Bool.eq_false_iff.mpr ready
    simp only [emit, if_neg ready]
    exact BlockContinues.branch (flag := false) (by simpa only [notReady] using tested) rfl .nil

theorem collect_loop (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap)
    (remaining : List String) (inside : ∀ vertex ∈ remaining, vertex ∈ state.vertices)
    (collected : List String) (env : HeapCalls.Env)
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.setStepLoop "v" roots.nodes (stringSet state.vertices) (stringSet remaining)
            (body heap.length) [] :: following) env, stack, heap ++ [.list (values collected)]⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (collected ++ remaining.filter (eligible state)) := by
  have nodesInside := (List.getElem?_eq_some_iff.mp represented.nodesFound).1
  have nodesFound : (heap ++ [.list (values collected)])[roots.nodes]? =
      some (.set (stringSet state.vertices)) := by
    rw [List.getElem?_append_left nodesInside]
    exact represented.nodesFound
  cases remainingEq : remaining with
  | nil =>
      refine ⟨collected, env, used, .next ?_ .refl, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, stringSet, nodesFound]
      · simp
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let vertex := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      let boundEnv := setLocal env "v" (text vertex)
      have selected : (stringSet (head :: tail))[choices used % (tail.length + 1)]? =
          some (.string vertex) := by
        simp only [stringSet, List.getElem?_map]
        rw [show choices used % (tail.length + 1) = index from rfl,
          List.getElem?_eq_getElem bound]
        rfl
      have selected' : (Constant.string head :: tail.map Constant.string)[
          choices used % (tail.length + 1)]? = some (.string vertex) := selected
      have erased : (Constant.string head :: tail.map Constant.string).eraseIdx
          (choices used % (tail.length + 1)) = rest.map Constant.string :=
        List.eraseIdx_map Constant.string (head :: tail) index
      have emitted := collect_one roots state heap represented env getDegrees vertex
        (inside vertex (by rw [remainingEq]; exact List.getElem_mem bound)) collected
      obtain ⟨result, finalEnv, finalUsed, completed, contents⟩ :=
        collect_loop choices roots state heap represented rest
          (fun value member => inside value (by
            rw [remainingEq]
            exact List.mem_of_mem_eraseIdx member))
          (emit state collected vertex) boundEnv (by simpa [boundEnv, setLocal] using getDegrees)
          following anchors stack (used + 1)
      refine ⟨result, finalEnv, finalUsed, .next (following :=
        ⟨⟨.block (body heap.length ++ [.setStepLoop "v" roots.nodes (stringSet state.vertices)
            (stringSet rest) (body heap.length) []] ++ following) boundEnv,
          stack, heap ++ [.list (values collected)]⟩, used + 1⟩) ?_ ?_, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, nodesFound,
          selected', erased, stringSet, boundEnv,
          body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]
      · exact (SetCalls.Reaches.of_heap (emitted
          (.setStepLoop "v" roots.nodes (stringSet state.vertices) (stringSet rest)
            (body heap.length) [] :: following) anchors stack) choices (used + 1)).trans completed
      · apply contents.trans
        have filtered := (List.getElem_cons_eraseIdx_perm bound).filter (eligible state)
        have appended := filtered.append_left collected
        by_cases ready : eligible state vertex = true
        · simpa [emit, ready, vertex, rest, List.append_assoc] using appended
        · simpa [emit, ready, vertex, rest] using appended
termination_by remaining.length
decreasing_by
  simp only [remainingEq]
  change ((head :: tail).eraseIdx index).length < (head :: tail).length
  rw [List.length_eraseIdx_of_lt bound]
  simp

theorem expression_result (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.expr expression env, stack, heap⟩, used⟩
        ⟨⟨.value (.ref heap.length), stack, heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (state.vertices.filter (eligible state)) := by
  let inner := comprehensionEnv env [("v", .name "nodes", [condition])]
  have bound : inner "indeg" = some (.ref roots.degrees) := by
    simpa [inner, comprehensionEnv] using getDegrees
  obtain ⟨result, finalEnv, finalUsed, loop, contents⟩ :=
    collect_loop choices roots state heap represented state.vertices (fun _ member => member)
      [] inner bound [] anchors (.comprehensionResult heap.length :: stack) used
  have start := SetCalls.list_comp_set_start program env heap "nodes" "v" roots.nodes
    (stringSet state.vertices) getNodes represented.nodesFound (.name "v") [condition] []
    choices anchors stack used
  have finish : HeapCalls.Reaches program anchors
      ⟨.block [] finalEnv, .comprehensionResult heap.length :: stack,
        heap ++ [.list (values result)]⟩
      ⟨.value (.ref heap.length), stack, heap ++ [.list (values result)]⟩ :=
    .next rfl (.next rfl .refl)
  refine ⟨result, finalUsed, start.trans (loop.trans
    (SetCalls.Reaches.of_heap finish choices finalUsed)), ?_⟩
  simpa using contents

/-- The actual first statement allocates the complete ready queue, preserving surrounding locals. -/
theorem initial_result (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.block (topoCode.take 1 ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following (setLocal env "heap" (.ref heap.length)), stack,
          heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (state.vertices.filter (eligible state)) := by
  obtain ⟨result, finalUsed, evaluated, contents⟩ :=
    expression_result choices roots state heap represented env getNodes getDegrees anchors
      (.assign "heap" following env :: stack) used
  refine ⟨result, finalUsed, ?_, contents⟩
  have enter : HeapCalls.Reaches program anchors
      ⟨.block (topoCode.take 1 ++ following) env, stack, heap⟩
      ⟨.expr expression env, .assign "heap" following env :: stack, heap⟩ := .next rfl .refl
  have finish : HeapCalls.Reaches program anchors
      ⟨.value (.ref heap.length), .assign "heap" following env :: stack,
        heap ++ [.list (values result)]⟩
      ⟨.block following (setLocal env "heap" (.ref heap.length)), stack,
        heap ++ [.list (values result)]⟩ := .next rfl .refl
  exact (SetCalls.Reaches.of_heap enter choices used).trans
    (evaluated.trans (SetCalls.Reaches.of_heap finish choices finalUsed))

theorem ready_contents (state : Storage) (result : List String)
    (contents : result.Perm (state.vertices.filter (eligible state))) (vertex : String) :
    vertex ∈ result ↔ vertex ∈ state.vertices ∧ Ready vertex state.edges := by
  rw [contents.mem_iff]
  simp [eligible, indegree_zero_iff]

end solvers.«79».ready
