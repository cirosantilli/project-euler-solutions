import solvers.«61.prefix_store.source»

set_option Elab.async false

namespace solvers.«61».prefix_setup
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open dfs (numbers)
open data.«project-euler-statements».data.py.project_euler_statements.p61.table (encode lookup_encode encode_put)
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

def familyExpression : Expr := .subscript (.name "nums_by_type") (.name "t")
def valueExpression : Expr := .call (.name "build_prefix_map") [familyExpression]
def body (address : Nat) : List Stmt := [.comprehensionStore address (.name "t") valueExpression]
def generators : List (Expr × Expr × List Expr) := [(.name "t", .name "types", [])]
def expression : Expr := .dictComp (.name "t") valueExpression generators

variable {base heap numsBase : Mutable.Heap} {processed kinds : List Int}

theorem family_evaluate (nums : NumberStore numsBase base kinds) (store : Store base heap processed)
    (kind : Int) (member : kind ∈ kinds) (env : HeapCalls.Env)
    (getNums : env "nums_by_type" = some (.ref numsBase.length)) (getKind : env "t" = some (number kind))
    (globals : Mutable.Globals := []) :
    ∃ input, Evaluates program env familyExpression heap (.ref input) heap globals ∧
      heap[input]? = some (.list (numbers (PolygonalFamilies.family kind))) := by
  obtain ⟨input, _, looked, found⟩ := nums.present kind member
  obtain ⟨extra, current⟩ := store.extension
  have tableFound : heap[numsBase.length]? = some (.dict (encode nums.entries)) := by
    rw [current]
    exact HeapCollection.prefix_found _ _ _ _ nums.table
  refine ⟨input, ?_, ?_⟩
  · apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getNums) (Evaluates.lookup getKind)
    apply Mutable.getItem_dictionary heap numsBase.length _ tableFound (.int kind)
    rw [lookup_encode, looked]
    rfl
  · rw [current]
    exact HeapCollection.prefix_found _ _ _ _ found

theorem store_result (nums : NumberStore numsBase base kinds) (store : Store base heap processed)
    (kind : Int) (member : kind ∈ kinds) (env : HeapCalls.Env)
    (getNums : env "nums_by_type" = some (.ref numsBase.length)) (getKind : env "t" = some (number kind))
    (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_store : Store base finalHeap (processed ++ [kind])),
      BlockContinues program env (body base.length) heap env finalHeap globals := by
  obtain ⟨input, looked, found⟩ := family_evaluate nums store kind member env getNums getKind globals
  obtain ⟨after, child, built⟩ := build_prefix_map.result (PolygonalFamilies.family kind) heap input found globals
  have evaluated : Evaluates program env valueExpression heap (.ref heap.length) after globals :=
    Evaluates.call (.cons looked .nil) built
  have tableFound : after[base.length]? = some (.dict (encode store.entries)) := by
    obtain ⟨extra, rfl⟩ := child.extension
    exact HeapCollection.prefix_found _ _ _ _ store.table
  refine ⟨write base after store.entries kind heap.length, store.insert_child kind child, ?_⟩
  apply BlockContinues.comprehension_store (.cons (Evaluates.lookup getKind) (.cons evaluated .nil))
  simpa only [write, encode_put] using
    Mutable.setItem_dictionary _ _ _ tableFound (.int kind) (.ref heap.length)

theorem loop (nums : NumberStore numsBase base kinds) (remaining : List Int) (position : Nat)
    (suffix : kinds.drop position = remaining) (store : Store base heap processed)
    (input : Nat) (found : base[input]? = some (.list (numbers kinds))) (env : HeapCalls.Env)
    (getNums : env "nums_by_type" = some (.ref numsBase.length)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env) (_store : Store base finalHeap (processed ++ remaining)),
      BlockContinues program env [.sequenceStepLoop "t" (.ref input) position (body base.length) []]
        heap finalEnv finalHeap globals := by
  have source (currentHeap : Mutable.Heap) (processed : List Int) (store : Store base currentHeap processed) :
      currentHeap[input]? = some (.list (numbers kinds)) := by
    obtain ⟨extra, rfl⟩ := store.extension
    exact HeapCollection.prefix_found _ _ _ _ found
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, env, ?_, ?_⟩
      · simpa only [List.append_nil] using store
      · apply BlockContinues.sequence_done (items := numbers kinds)
          (by simp [sequenceValue, Mutable.sequence, source heap processed store])
        · have done : kinds[position]? = none := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, done]
        · exact .nil
  | cons kind rest ih =>
      have item : kinds[position]? = some kind := by rw [← List.head?_drop, suffix]; rfl
      let bound := setLocal env "t" (number kind)
      have boundNums : bound "nums_by_type" = some (.ref numsBase.length) := by simpa [bound, setLocal] using getNums
      obtain ⟨middle, middleStore, generated⟩ := store_result nums store kind (List.mem_of_getElem? item) bound boundNums rfl globals
      obtain ⟨finalHeap, finalEnv, finalStore, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) middleStore bound boundNums
      refine ⟨finalHeap, finalEnv, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finalStore
      · apply BlockContinues.sequence_step (items := numbers kinds) (value := number kind)
          (by simp [sequenceValue, Mutable.sequence, source heap processed store])
        · simp [numbers, List.getElem?_map, item]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using generated.append executed

/-- Every source callback runs the verified bucket builder on the corresponding complete family. -/
theorem result (nums : NumberStore numsBase base kinds) (input : Nat)
    (found : base[input]? = some (.list (numbers kinds))) (env : HeapCalls.Env)
    (getTypes : env "types" = some (.ref input)) (getNums : env "nums_by_type" = some (.ref numsBase.length))
    (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_store : Store base finalHeap kinds),
      Evaluates program env expression base (.ref base.length) finalHeap globals := by
  let masked := comprehensionTargetsEnv env generators
  have maskedNums : masked "nums_by_type" = some (.ref numsBase.length) := by
    simpa [masked, generators, comprehensionTargetsEnv, targetLocals] using getNums
  obtain ⟨finalHeap, finalEnv, store, executed⟩ := loop nums kinds 0 rfl (Store.initial base) input found masked maskedNums globals
  refine ⟨finalHeap, by simpa only [List.nil_append] using store, ?_⟩
  apply Evaluates.dict_comp_sequence (receiver := .ref input) (items := numbers kinds)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getTypes) (by simp [Mutable.sequence, found])
  simpa only [masked, generators, comprehensionTargetName, comprehensionTargetBody, comprehensionFilters,
    List.foldr_nil, comprehensionTargetsBody, body, sequenceSource] using executed

end solvers.«61».prefix_setup
