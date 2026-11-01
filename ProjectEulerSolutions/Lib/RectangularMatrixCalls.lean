import FormalSW

/-! Heap-preserving rectangularity checks for nested integer lists. -/
namespace ProjectEulerSolutions.RectangularMatrixCalls
open FormalSW.PyAst
set_option Elab.async false
variable {program : Module}
abbrev encode (values : List Int) : List Mutable.Value := values.map (fun value => .scalar (.int value))
abbrev MatrixRows := List (Nat × List Int)
abbrev lengthTest : Expr :=
  .compare (.builtinCall .len [.name "row"]) [(.notEq, .name "width")]
abbrev shapeBody : List Stmt :=
  [.ifThen lengthTest [.return (.constant (.bool false))] []]

theorem rows_check (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (width position : Nat) (env : HeapCalls.Env) (localWidth : env "width" = some (.scalar (.int width))) :
    ((∀ (i : Nat) (valid : i < rows.length), position ≤ i → rows[i].2.length = width) ∧
      ∃ finalEnv, HeapCalls.BlockContinues program env
        [.sequenceStepLoop "row" (.ref address) position shapeBody []] heap finalEnv heap) ∨
    (∀ following, HeapCalls.BlockReturns program env
      (.sequenceStepLoop "row" (.ref address) position shapeBody [] :: following)
      heap (.scalar (.bool false)) heap) := by
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by simp [Mutable.sequence, found]
  by_cases active : position < rows.length
  · let pair := rows[position]
    let boundEnv := HeapCalls.setLocal env "row" (.ref pair.1)
    have item : (rows.map (fun pair => Mutable.Value.ref pair.1))[position]? = some (.ref pair.1) := by
      simp [List.getElem?_eq_getElem active, pair]
    have savedSize : boundEnv "width" = some (.scalar (.int width)) := by
      simpa [boundEnv, HeapCalls.setLocal] using localWidth
    have tested : HeapCalls.Evaluates program boundEnv lengthTest heap
        (.scalar (.bool (decide (pair.2.length ≠ width)))) heap := by
      apply HeapCalls.Evaluates.compare (av := .scalar (.int pair.2.length)) (middle := heap)
      · exact HeapCalls.Evaluates.builtin (.cons (.lookup rfl) .nil)
          (by simp [HeapCalls.primitive, Mutable.length, reads pair (List.getElem_mem active), encode])
      · exact .lookup savedSize
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
    by_cases good : pair.2.length = width
    · have step : HeapCalls.BlockContinues program boundEnv shapeBody heap boundEnv heap :=
        HeapCalls.BlockContinues.branch (flag := false) (by simpa [good] using tested) rfl HeapCalls.BlockContinues.nil
      rcases rows_check heap address rows found reads width (position + 1) boundEnv savedSize with
        ⟨goodRows, finalEnv, continued⟩ | rejected
      · left
        refine ⟨?_, finalEnv, ?_⟩
        · intro i valid after
          by_cases same : i = position
          · subst i
            exact good
          · exact goodRows i valid (by omega)
        · apply HeapCalls.BlockContinues.sequence_step (source := .ref address) sequence item
          simpa [shapeBody, Calls.loopContinuation, Calls.loopControl] using step.append continued
      · right
        intro following
        apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
        simpa [shapeBody, Calls.loopContinuation, Calls.loopControl] using step.returns (rejected following)
    · right
      intro following
      apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
      exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [good] using tested) rfl
        (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
  · left
    refine ⟨by intros; omega, env, ?_⟩
    exact HeapCalls.BlockContinues.sequence_done (source := .ref address) sequence
      (by simp [List.getElem?_eq_none_iff.mpr (Nat.le_of_not_gt active)]) HeapCalls.BlockContinues.nil
termination_by rows.length - position
decreasing_by omega


def Shape (rows : MatrixRows) : Prop :=
  ∀ (index : Nat) (valid : index < rows.length), rows[index].2.length = (rows.headD (0, [])).2.length

abbrev arguments : List Arg :=
  [⟨"matrix", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩]

abbrev shapeCode : List Stmt :=
  [.ifThen (.unaryOp .not (.name "matrix")) [.return (.constant (.bool true))] [],
   .assign [.name "width"] (.builtinCall .len [.subscript (.name "matrix") (.constant (.int 0))]),
   .forIn (.name "row") (.name "matrix") shapeBody [],
   .return (.constant (.bool true))]

theorem is_rectangular_result
    (source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode))
    (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    (Shape rows ∧ HeapCalls.ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) ∨
      HeapCalls.ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool false)) heap := by
  let env := HeapCalls.argumentEnv arguments [.ref address]
  have tested : HeapCalls.Evaluates program env (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool (decide (rows = [])))) heap := by
    apply HeapCalls.Evaluates.unary (HeapCalls.Evaluates.lookup rfl)
    simp [HeapCalls.unary, Mutable.truth, found]
    cases rows <;> rfl
  cases rows with
  | nil =>
      left
      refine ⟨by simp [Shape], ?_⟩
      apply HeapCalls.BlockReturns.function source rfl
      exact HeapCalls.BlockReturns.branch tested rfl (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
  | cons first rest =>
      let sized := HeapCalls.setLocal env "width" (.scalar (.int first.2.length))
      have firstValue : HeapCalls.Evaluates program env
          (.subscript (.name "matrix") (.constant (.int 0))) heap (.ref first.1) heap := by
        apply HeapCalls.Evaluates.subscript (by intros a b c impossible; cases impossible)
          (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        simp [Mutable.getItem, found, Mutable.index, Except.map]
      have widthValue : HeapCalls.Evaluates program env
          (.builtinCall .len [.subscript (.name "matrix") (.constant (.int 0))]) heap
          (.scalar (.int first.2.length)) heap := by
        apply HeapCalls.Evaluates.builtin (.cons firstValue .nil)
        simp [HeapCalls.primitive, Mutable.length, reads first (by simp), encode]
      have sequence : Mutable.sequence (.ref address) heap =
          some (.ok ((first :: rest).map (fun pair => .ref pair.1)), heap) := by
        simp [Mutable.sequence, found]
      rcases rows_check (program := program) heap address (first :: rest) found reads first.2.length 0 sized rfl with
        ⟨good, finalEnv, loop⟩ | rejected
      · left
        refine ⟨fun index valid => good index valid (by omega), ?_⟩
        apply HeapCalls.BlockReturns.function source rfl
        apply HeapCalls.BlockReturns.branch (flag := false) tested rfl
        apply HeapCalls.BlockReturns.assign widthValue
        exact (HeapCalls.BlockContinues.for_sequence (by intros expressions impossible; cases impossible)
          (HeapCalls.Evaluates.lookup rfl) sequence loop).returns
          (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
      · right
        apply HeapCalls.BlockReturns.function source rfl
        apply HeapCalls.BlockReturns.branch (flag := false) tested rfl
        apply HeapCalls.BlockReturns.assign widthValue
        exact HeapCalls.BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
          (HeapCalls.Evaluates.lookup rfl) sequence (rejected [.return (.constant (.bool true))])

/-- Decoding preserves row identities, including repeated references to one row. -/
theorem decode_rows (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ rows : MatrixRows,
      heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
      ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)) := by
  classical
  obtain ⟨addresses, found, rowTypes⟩ := Mutable.Value.lists_of_hasType typed
  have decoded : ∀ row ∈ addresses, ∃ values : List Int, heap[row]? = some (.list (encode values)) :=
    fun row member => Mutable.Value.ints_of_hasType (rowTypes row member)
  let values := fun row member => (decoded row member).choose
  have contents := fun row member => (decoded row member).choose_spec
  let rows := addresses.attach.map (fun entry => (entry.val, values entry.val entry.property))
  have same : rows.map (fun pair => Mutable.Value.ref pair.1) = addresses.map Mutable.Value.ref := by
    simp [rows, List.map_map]
  refine ⟨rows, by simpa [same] using found, ?_⟩
  intro pair member
  obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
  exact contents entry.val entry.property

abbrev matrixEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "matrix" then some (.ref address) else none

abbrev shapeExpression : Expr := .call (.name "is_rectangular") [.name "matrix"]

theorem condition_result
    (source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode))
    (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ flag, HeapCalls.Evaluates program (matrixEnv address) shapeExpression heap (.scalar (.bool flag)) heap ∧
      (flag = true → ∃ rows : MatrixRows,
        heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
        (∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) ∧ Shape rows ∧
        HeapCalls.ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) := by
  obtain ⟨rows, found, reads⟩ := decode_rows heap address typed
  rcases is_rectangular_result source heap address rows found reads with ⟨shape, returned⟩ | returned
  · exact ⟨true, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) returned,
      fun _ => ⟨rows, found, reads, shape, returned⟩⟩
  · exact ⟨false, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) returned, by simp⟩

theorem domain_of_condition
    (source : findFunctionIn program.body "is_rectangular" = some (arguments, shapeCode))
    (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (matrixEnv address) shapeExpression heap) :
    ∃ rows : MatrixRows,
      heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
      (∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) ∧ Shape rows ∧
      HeapCalls.ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap := by
  obtain ⟨flag, evaluated, domain⟩ := condition_result source heap address typed
  exact domain (by simpa using evaluated.condition_iff.mp requires)

end ProjectEulerSolutions.RectangularMatrixCalls
