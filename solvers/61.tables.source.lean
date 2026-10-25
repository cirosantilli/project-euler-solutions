import solvers.«61.families.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.table.source»

set_option Elab.async false

namespace solvers.«61».tables
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open data.«project-euler-statements».data.py.project_euler_statements.p61.table (Store storedHeap encode_put)
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private def numbers (values : List Int) : List Mutable.Value := values.map number

def valueExpression : Expr := .call (.name "generate_4digit_polygonals") [.name "t"]
def body (address : Nat) : List Stmt := [.comprehensionStore address (.name "t") valueExpression]
def generators : List (Expr × Expr × List Expr) := [(.name "t", .name "types", [])]
def expression : Expr := .dictComp (.name "t") valueExpression generators

variable {base heap : Mutable.Heap} {processed : List Int}

theorem source_found (store : Store base heap processed) (input : Nat) (values : List Int)
    (found : base[input]? = some (.list (numbers values))) :
    heap[input]? = some (.list (numbers values)) := by
  obtain ⟨extra, rfl⟩ := store.extension
  exact HeapCollection.prefix_found _ _ _ _ found

theorem store_result (store : Store base heap processed) (sides : Int) (valid : 3 ≤ sides)
    (env : HeapCalls.Env) (getType : env "t" = some (number sides)) (globals : Mutable.Globals := []) :
    BlockContinues program env (body base.length) heap env (storedHeap base heap store.entries sides) globals := by
  have evaluated : Evaluates program env valueExpression heap (.ref heap.length)
      (heap ++ [.list (numbers (PolygonalFamilies.family sides))]) globals :=
    Evaluates.call (.cons (Evaluates.lookup getType) .nil) (generate_4digit_polygonals.result sides valid heap globals)
  have tableFound := HeapCollection.prefix_found heap [.list (numbers (PolygonalFamilies.family sides))]
    base.length _ store.table
  apply BlockContinues.comprehension_store (.cons (Evaluates.lookup getType) (.cons evaluated .nil))
  simpa only [storedHeap, encode_put] using
    Mutable.setItem_dictionary _ _ _ tableFound (.int sides) (.ref heap.length)

theorem loop (remaining values : List Int) (position : Nat) (suffix : values.drop position = remaining)
    (valid : ∀ sides ∈ remaining, (3 : Int) ≤ sides) (store : Store base heap processed)
    (input : Nat) (found : base[input]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env) (_store : Store base finalHeap (processed ++ remaining)),
      BlockContinues program env [.sequenceStepLoop "t" (.ref input) position (body base.length) []]
        heap finalEnv finalHeap globals := by
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, env, ?_, ?_⟩
      · simpa only [List.append_nil] using store
      · apply BlockContinues.sequence_done (items := numbers values)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have done : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, done]
        · exact .nil
  | cons sides rest ih =>
      let bound := setLocal env "t" (number sides)
      have generated := store_result store sides (valid sides (by simp)) bound rfl globals
      obtain ⟨finalHeap, finalEnv, finalStore, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons])
        (by intro side member; exact valid side (by simp [member])) (store.add sides) bound
      refine ⟨finalHeap, finalEnv, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finalStore
      · apply BlockContinues.sequence_step (items := numbers values) (value := number sides)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have item : values[position]? = some sides := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, item]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using generated.append executed

/-- The source comprehension builds a complete family dictionary for any list of valid type identifiers. -/
theorem result (values : List Int) (valid : ∀ side ∈ values, (3 : Int) ≤ side)
    (heap : Mutable.Heap) (input : Nat) (found : heap[input]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getTypes : env "types" = some (.ref input)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_store : Store heap finalHeap values),
      Evaluates program env expression heap (.ref heap.length) finalHeap globals := by
  obtain ⟨finalHeap, finalEnv, store, executed⟩ := loop values values 0 rfl valid (Store.initial heap)
    input found (comprehensionTargetsEnv env generators) globals
  refine ⟨finalHeap, by simpa only [List.nil_append] using store, ?_⟩
  apply Evaluates.dict_comp_sequence (receiver := .ref input) (items := numbers values)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getTypes) (by simp [Mutable.sequence, found])
  simpa only [generators, comprehensionTargetName, comprehensionTargetBody, comprehensionFilters,
    List.foldr_nil, comprehensionTargetsBody, body, sequenceSource] using executed

end solvers.«61».tables
