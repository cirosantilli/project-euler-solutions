import solvers.«62.store.source»
import ProjectEulerSolutions.Lib.CubicSearch

/-! Complete loop control for the cubic-permutation source search. -/
set_option Elab.async false
namespace solvers.«62».search
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.CubicGroups ProjectEulerSolutions.CubicSearch
private abbrev program := Solvers.P62Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev SearchState := ProjectEulerSolutions.CubicSearch.State

def cubeExpression : Expr := .binOp (.binOp (.name "n") .mult (.name "n")) .mult (.name "n")
def widthExpression : Expr := .builtinCall .len [.builtinCall .str [.name "cube"]]
def prepareCode : List Stmt := [.assign [.name "cube"] cubeExpression, .assign [.name "length"] widthExpression]
def widthTest : Expr := .compare (.name "length") [(.gt, .name "current_len")]
def widthCode : List Stmt := [.ifThen widthTest groups.widthBody []]
def incrementCode : List Stmt := [.augAssign (.name "n") .add (.constant (.int 1))]
def loop : Stmt := .whileDo (.constant (.bool true)) groups.iteration []
theorem iteration_source : groups.iteration = prepareCode ++ widthCode ++ groups.groupingCode ++ incrementCode := rfl
theorem continuation : Calls.loopContinuation groups.iteration loop = loop := rfl

def prepared (env : HeapCalls.Env) (n : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "cube" (number (cube n))) "length" (number (width n))

theorem cube_evaluate (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) : Evaluates program env cubeExpression heap (number (cube n)) heap := by
  have square : Evaluates program env (.binOp (.name "n") .mult (.name "n")) heap (number (n * n : Nat)) heap := by
    apply Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getN)
    simp [Mutable.binary, scalarBinary]
  apply Evaluates.binary square (Evaluates.lookup getN)
  simp [cube, Mutable.binary, scalarBinary]

theorem width_evaluate (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getCube : env "cube" = some (number (cube n))) :
    Evaluates program env widthExpression heap (number (width n)) heap := by
  have text : Evaluates program env (.builtinCall .str [.name "cube"]) heap
      (.scalar (.string (toString (cube n : Int)))) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getCube) .nil) rfl
  apply Evaluates.builtin (.cons text .nil)
  change (some (.ok (number (toString (cube n : Int)).length), heap) :
    Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
  have length : (toString (cube n : Int)).length = width n := by
    rw [← String.length_toList, ProjectEulerSolutions.CubicPermutations.text_decimal]
    simp [width]
  rw [length]

theorem prepare_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    BlockContinues program env prepareCode heap (prepared env n) heap :=
  (BlockContinues.assign (cube_evaluate n heap env getN)).append
    (BlockContinues.assign (width_evaluate n heap (setLocal env "cube" (number (cube n))) (by simp [setLocal])))

theorem test_evaluate (n currentWidth : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLength : env "length" = some (number (width n)))
    (getCurrent : env "current_len" = some (number currentWidth)) :
    Evaluates program env widthTest heap (flag (decide (currentWidth < width n))) heap := by
  apply Evaluates.compare (Evaluates.lookup getLength) (Evaluates.lookup getCurrent)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem increment_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    BlockContinues program env incrementCode heap (setLocal env "n" (number (n + 1 : Nat))) heap := by
  apply BlockContinues.augAssign getN Evaluates.constant
  simp [inplace, Mutable.binary, scalarBinary]

structure Inputs (env : HeapCalls.Env) (target : Int) (state : SearchState) (dictionary : Nat) : Prop where
  n : env "n" = some (number state.next)
  current : env "current_len" = some (number state.currentWidth)
  groups : env "groups" = some (.ref dictionary)
  k : env "k" = some (number target)

structure Invariant (heap : Mutable.Heap) (env : HeapCalls.Env) (target : Int) (state : SearchState)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat) : Prop where
  inputs : Inputs env target state dictionary
  table : store.Store heap dictionary cells
  logical : store.logical cells = state.groups
  dictionaryFresh : floor ≤ dictionary
  cellsFresh : store.Above floor cells

def Inputs.prepared {env : HeapCalls.Env} {target : Int} {state : SearchState} {dictionary : Nat}
    (inputs : Inputs env target state dictionary) : Inputs (prepared env state.next) target state dictionary where
  n := by simpa [prepared, setLocal] using inputs.n
  current := by simpa [prepared, setLocal] using inputs.current
  groups := by simpa [prepared, setLocal] using inputs.groups
  k := by simpa [prepared, setLocal] using inputs.k

theorem group_increment (target : Int) (state : SearchState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor)
    (getCube : env "cube" = some (number (cube state.next))) :
    ∃ finalHeap finalEnv finalCells,
      BlockContinues program env (groups.groupingCode ++ incrementCode) heap finalEnv finalHeap ∧
      Invariant finalHeap finalEnv target (advance state state.currentWidth state.groups) dictionary finalCells floor ∧
      finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length := by
  obtain ⟨finalHeap, finalCells, grouped, table, logical, above, kept, grows⟩ :=
    store.grouping_preserves (cube state.next) heap env dictionary cells invariant.table invariant.inputs.groups
      getCube floor invariant.dictionaryFresh invariant.cellsFresh
  let groupedEnv := setLocal env "sig" (.ref (heap.length + 2))
  let finalEnv := setLocal groupedEnv "n" (number (state.next + 1 : Nat))
  have incremented := increment_result state.next finalHeap groupedEnv (by simpa [groupedEnv, setLocal] using invariant.inputs.n)
  refine ⟨finalHeap, finalEnv, finalCells, grouped.append incremented, ?_, kept, grows⟩
  constructor
  · constructor
    · simp [finalEnv, advance, setLocal]
    · simpa [finalEnv, groupedEnv, advance, setLocal] using invariant.inputs.current
    · simpa [finalEnv, groupedEnv, setLocal] using invariant.inputs.groups
    · simpa [finalEnv, groupedEnv, setLocal] using invariant.inputs.k
  · exact table
  · simpa only [advance, invariant.logical] using logical
  · exact invariant.dictionaryFresh
  · exact above

theorem same_width (target : Int) (state : SearchState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor)
    (notGrowing : ¬state.currentWidth < width state.next) :
    ∃ finalHeap finalEnv finalCells,
      BlockContinues program env groups.iteration heap finalEnv finalHeap ∧
      Invariant finalHeap finalEnv target (advance state state.currentWidth state.groups) dictionary finalCells floor ∧
      finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length := by
  let readyEnv := prepared env state.next
  have readyInvariant : Invariant heap readyEnv target state dictionary cells floor :=
    ⟨invariant.inputs.prepared, invariant.table, invariant.logical, invariant.dictionaryFresh, invariant.cellsFresh⟩
  have preparedRun := prepare_result state.next heap env invariant.inputs.n
  have tested := test_evaluate state.next state.currentWidth heap readyEnv (by simp [readyEnv, prepared, setLocal])
    readyInvariant.inputs.current
  have skipped : BlockContinues program readyEnv widthCode heap readyEnv heap := by
    apply BlockContinues.branch (flag := false) (by simpa only [notGrowing, decide_false] using tested) rfl
    exact .nil
  obtain ⟨finalHeap, finalEnv, finalCells, executed, finalInvariant, kept, grows⟩ :=
    group_increment target state heap readyEnv dictionary cells floor readyInvariant (by simp [readyEnv, prepared, setLocal])
  refine ⟨finalHeap, finalEnv, finalCells, ?_, finalInvariant, kept, grows⟩
  rw [iteration_source]
  simpa only [List.append_assoc] using (preparedRun.append skipped).append executed

theorem new_width (target : Int) (state : SearchState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor)
    (growing : state.currentWidth < width state.next) (empty : matching target state.groups = []) :
    ∃ finalHeap finalEnv finalCells,
      BlockContinues program env groups.iteration heap finalEnv finalHeap ∧
      Invariant finalHeap finalEnv target (advance state (width state.next) []) dictionary finalCells floor ∧
      finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length := by
  let readyEnv := prepared env state.next
  let resetEnv := setLocal (setLocal readyEnv "candidates" (.ref heap.length)) "current_len" (number (width state.next))
  let resetState : SearchState := ⟨state.next, width state.next, []⟩
  have inputs := invariant.inputs.prepared
  have preparedRun := prepare_result state.next heap env invariant.inputs.n
  have tested := test_evaluate state.next state.currentWidth heap readyEnv (by simp [readyEnv, prepared, setLocal]) inputs.current
  obtain ⟨garbage, resetRun, resetTable, resetKept, resetGrows⟩ := store.width_continues target (width state.next)
    heap readyEnv dictionary cells invariant.table inputs.groups inputs.k (by simp [readyEnv, prepared, setLocal])
    (by simpa only [invariant.logical] using empty) floor invariant.dictionaryFresh
  have changed : BlockContinues program readyEnv widthCode heap resetEnv (store.resetHeap heap dictionary garbage) :=
    BlockContinues.branch (flag := true) (by simpa only [growing, decide_true] using tested) rfl resetRun
  have resetInvariant : Invariant (store.resetHeap heap dictionary garbage) resetEnv target resetState dictionary [] floor := by
    constructor
    · constructor
      · simpa [resetEnv, resetState, setLocal] using inputs.n
      · simp [resetEnv, resetState, setLocal]
      · simpa [resetEnv, setLocal] using inputs.groups
      · simpa [resetEnv, setLocal] using inputs.k
    · exact resetTable
    · rfl
    · exact invariant.dictionaryFresh
    · simp [store.Above]
  obtain ⟨finalHeap, finalEnv, finalCells, executed, finalInvariant, kept, grows⟩ :=
    group_increment target resetState (store.resetHeap heap dictionary garbage) resetEnv dictionary [] floor resetInvariant
      (by simp [resetEnv, readyEnv, resetState, prepared, setLocal])
  refine ⟨finalHeap, finalEnv, finalCells, ?_, finalInvariant, kept.trans resetKept, resetGrows.trans grows⟩
  rw [iteration_source]
  simpa only [List.append_assoc] using (preparedRun.append changed).append executed

/-- Explicit return is stable under every pending source continuation. -/
theorem width_return_following (target : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (cells : List store.Cell) (table : store.Store heap dictionary cells)
    (getGroups : env "groups" = some (.ref dictionary)) (getK : env "k" = some (number target))
    (first : Int) (rest : List Int) (selected : matching target (store.logical cells) = first :: rest) :
    ∃ garbage, ∀ following,
      BlockReturns program env (groups.widthBody ++ following) heap (number (rest.foldl min first))
        (ProjectEulerSolutions.HeapCollection.working heap ((matching target (store.logical cells)).map number) garbage) := by
  obtain ⟨garbage, collected⟩ := candidates.source_result target heap env dictionary (cells.map store.Cell.candidate)
    table.candidates getGroups getK
  simp only [store.matching_candidates] at collected
  refine ⟨garbage, ?_⟩
  intro following
  rw [candidates.winner_source, List.append_assoc, List.append_assoc]
  apply collected.returns
  apply candidates.winner first rest _ _ heap.length (by simp [setLocal])
  simpa only [selected] using
    ProjectEulerSolutions.HeapCollection.found heap ((matching target (store.logical cells)).map number) garbage

theorem return_at_width (target : Int) (state : SearchState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor)
    (growing : state.currentWidth < width state.next) (first : Int) (rest : List Int)
    (selected : matching target state.groups = first :: rest) :
    ∃ finalHeap, (∀ following, BlockReturns program env (groups.iteration ++ following) heap
      (number (rest.foldl min first)) finalHeap) ∧ finalHeap.take floor = heap.take floor := by
  let readyEnv := prepared env state.next
  have inputs := invariant.inputs.prepared
  have preparedRun := prepare_result state.next heap env invariant.inputs.n
  have tested := test_evaluate state.next state.currentWidth heap readyEnv (by simp [readyEnv, prepared, setLocal]) inputs.current
  obtain ⟨garbage, returned⟩ := width_return_following target heap readyEnv dictionary cells invariant.table
    inputs.groups inputs.k first rest (by simpa only [invariant.logical] using selected)
  refine ⟨ProjectEulerSolutions.HeapCollection.working heap ((matching target (store.logical cells)).map number) garbage, ?_, ?_⟩
  · intro following
    rw [iteration_source]
    simp only [List.append_assoc]
    apply preparedRun.returns
    apply BlockReturns.branch (flag := true) (by simpa only [growing, decide_true] using tested) rfl
    exact returned (groups.groupingCode ++ incrementCode ++ following)
  · apply List.take_append_of_le_length
    have bound := (List.getElem?_eq_some_iff.mp invariant.table.found).1
    have fresh := invariant.dictionaryFresh
    omega

def Outcome (target : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary floor : Nat) : Sum Int SearchState → Prop
  | .inl value => ∃ finalHeap, (∀ following, BlockReturns program env (groups.iteration ++ following) heap (number value) finalHeap) ∧
      finalHeap.take floor = heap.take floor
  | .inr state => ∃ finalHeap finalEnv finalCells,
      BlockContinues program env groups.iteration heap finalEnv finalHeap ∧
      Invariant finalHeap finalEnv target state dictionary finalCells floor ∧
      finalHeap.take floor = heap.take floor ∧ heap.length ≤ finalHeap.length

theorem step_result (target : Int) (state : SearchState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor) :
    Outcome target heap env dictionary floor (ProjectEulerSolutions.CubicSearch.step target state) := by
  unfold ProjectEulerSolutions.CubicSearch.step
  by_cases growing : state.currentWidth < width state.next
  · rw [if_pos growing]
    cases selected : matching target state.groups with
    | nil => exact new_width target state heap env dictionary cells floor invariant growing selected
    | cons first rest => exact return_at_width target state heap env dictionary cells floor invariant growing first rest selected
  · rw [if_neg growing]
    exact same_width target state heap env dictionary cells floor invariant growing

/-- Every finite model witness executes the entire original while loop and preserves caller objects. -/
theorem loop_result (target : Int) (state : SearchState) (answer : Int)
    (witness : ProjectEulerSolutions.CubicSearch.Returns target state answer)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat) (cells : List store.Cell) (floor : Nat)
    (invariant : Invariant heap env target state dictionary cells floor) :
    ∃ finalHeap, BlockReturns program env [loop] heap (number answer) finalHeap ∧
      finalHeap.take floor = heap.take floor := by
  induction witness generalizing heap env cells with
  | done checked =>
      have executed := step_result target _ heap env dictionary cells floor invariant
      rw [checked] at executed
      obtain ⟨finalHeap, returned, kept⟩ := executed
      refine ⟨finalHeap, ?_, kept⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      change BlockReturns program env (groups.iteration ++ [loop]) heap _ finalHeap
      exact returned [loop]
  | next checked rest ih =>
      have executed := step_result target _ heap env dictionary cells floor invariant
      rw [checked] at executed
      obtain ⟨middleHeap, middleEnv, middleCells, bodyRun, middleInvariant, kept, grows⟩ := executed
      obtain ⟨finalHeap, returned, finalKept⟩ := ih middleHeap middleEnv middleCells middleInvariant
      refine ⟨finalHeap, ?_, finalKept.trans kept⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      change BlockReturns program env (groups.iteration ++ [loop]) heap _ finalHeap
      exact bodyRun.returns returned

def groupAnnotation : Expr := .subscript (.name "dict")
  (.tuple [.name "Signature", .subscript (.name "list") (.name "int")])
def initialization : List Stmt :=
  [.annAssign (.name "groups") groupAnnotation (some (.dict [])),
    .assign [.name "current_len"] (.constant (.int 1)), .assign [.name "n"] (.constant (.int 1))]
theorem code_source : executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 = initialization ++ [loop] := rfl
def initialHeap (caller : Mutable.Heap) : Mutable.Heap := caller ++ [.dict []]
def initialEnv (env : HeapCalls.Env) (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "groups" (.ref caller.length)) "current_len" (number 1)) "n" (number 1)

theorem initialize_result (caller : Mutable.Heap) (env : HeapCalls.Env) :
    BlockContinues program env initialization caller (initialEnv env caller) (initialHeap caller) := by
  have allocated : BlockContinues program env [.annAssign (.name "groups") groupAnnotation (some (.dict []))] caller
      (setLocal env "groups" (.ref caller.length)) (initialHeap caller) :=
    BlockContinues.annAssign (Evaluates.dictionary .nil)
  exact allocated.append ((BlockContinues.assign Evaluates.constant).append (BlockContinues.assign Evaluates.constant))

/-- The ordinary empty dictionary becomes a keyed live table during the actual first iteration. -/
theorem first_iteration (target : Int) (caller : Mutable.Heap) (env : HeapCalls.Env)
    (getK : env "k" = some (number target)) :
    ∃ finalHeap finalEnv finalCells,
      BlockContinues program (initialEnv env caller) groups.iteration (initialHeap caller) finalEnv finalHeap ∧
      Invariant finalHeap finalEnv target ready caller.length finalCells caller.length ∧
      finalHeap.take caller.length = caller := by
  let readyEnv := prepared (initialEnv env caller) 1
  let groupedEnv := setLocal readyEnv "sig" (.ref ((initialHeap caller).length + 2))
  let finalEnv := setLocal groupedEnv "n" (number 2)
  have initialized : Inputs (initialEnv env caller) target start caller.length := by
    constructor <;> simp [initialEnv, start, setLocal, getK]
  have preparedRun := prepare_result 1 (initialHeap caller) (initialEnv env caller) initialized.n
  have tested := test_evaluate 1 1 (initialHeap caller) readyEnv (by simp [readyEnv, prepared, setLocal])
    (by simp [readyEnv, prepared, initialEnv, setLocal])
  have skipped : BlockContinues program readyEnv widthCode (initialHeap caller) readyEnv (initialHeap caller) := by
    apply BlockContinues.branch (flag := false) (by simpa only [show width 1 = 1 from by decide] using tested) rfl
    exact .nil
  obtain ⟨grouped, table⟩ := store.grouping_initial 1 (initialHeap caller) readyEnv caller.length
    (by simp [readyEnv, prepared, initialEnv, setLocal]) (by simp [readyEnv, prepared, setLocal, cube])
    (by simp [initialHeap])
  have incremented := increment_result 1 (store.insertedHeap 1 (initialHeap caller) caller.length []) groupedEnv
    (by simp [groupedEnv, readyEnv, prepared, initialEnv, setLocal])
  refine ⟨store.insertedHeap 1 (initialHeap caller) caller.length [], finalEnv,
    [store.Cell.fresh 1 (initialHeap caller).length], ?_, ?_, ?_⟩
  · rw [iteration_source]
    simpa only [List.append_assoc] using ((preparedRun.append skipped).append grouped).append incremented
  · constructor
    · constructor
      · simp [finalEnv, ready, setLocal]
      · simp [finalEnv, groupedEnv, readyEnv, prepared, initialEnv, ready, setLocal]
      · simp [finalEnv, groupedEnv, readyEnv, prepared, initialEnv, setLocal]
      · simpa [finalEnv, groupedEnv, readyEnv, prepared, initialEnv, setLocal] using getK
    · exact table
    · rfl
    · exact Nat.le_refl _
    · apply store.Above.fresh (cells := []) (by simp [store.Above])
      simp [initialHeap]
  · have kept := store.insertedHeap_prefix 1 (initialHeap caller) caller.length [] caller.length
      (by simp [initialHeap]) (Nat.le_refl _)
    simpa [initialHeap] using kept

private def entry (target : Int) : HeapCalls.Env :=
  functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number target] []

/-- A finite witness executes the complete source function, including initialization and every loop step. -/
theorem result (target answer : Int) (caller : Mutable.Heap)
    (witness : ProjectEulerSolutions.CubicSearch.Returns target start answer) :
    ∃ finalHeap, ReturnsValue program "solve" [number target] caller (number answer) finalHeap ∧
      finalHeap.take caller.length = caller := by
  obtain ⟨middleHeap, middleEnv, middleCells, firstRun, invariant, kept⟩ :=
    first_iteration target caller (entry target) (by rfl)
  obtain ⟨finalHeap, returned, finalKept⟩ := loop_result target ready answer witness.ready_of_start
    middleHeap middleEnv caller.length middleCells caller.length invariant
  have wholeLoop : BlockReturns program (initialEnv (entry target) caller) [loop]
      (initialHeap caller) (number answer) finalHeap := by
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    change BlockReturns program (initialEnv (entry target) caller) (groups.iteration ++ [loop])
      (initialHeap caller) (number answer) finalHeap
    exact firstRun.returns returned
  refine ⟨finalHeap, ?_, finalKept.trans kept⟩
  apply BlockReturns.function rfl rfl
  change BlockReturns program (entry target) (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)
    caller (number answer) finalHeap
  rw [code_source]
  exact (initialize_result caller (entry target)).returns wholeLoop

end solvers.«62».search

namespace solvers.«62»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

theorem solve_source_result (target answer : Int) (caller : Mutable.Heap)
    (witness : ProjectEulerSolutions.CubicSearch.Returns target ProjectEulerSolutions.CubicSearch.start answer) :
    ∃ finalHeap, ReturnsValue Solvers.P62Ast.module "solve" [.scalar (.int target)] caller (.scalar (.int answer)) finalHeap ∧
      finalHeap.take caller.length = caller := search.result target answer caller witness

end solvers.«62»
