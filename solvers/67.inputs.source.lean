import «.formalsw».solvers.«67.py.ast»
import FormalSW
import ProjectEulerStatements.Lib.TriangleBuffers

namespace solvers.«67»
open FormalSW.PyAst
set_option Elab.async false
abbrev program := Solvers.P67Ast.module
abbrev encode (values : List Int) : List Mutable.Value := values.map (fun value => .scalar (.int value))
abbrev TriangleRows := List (Nat × List Int)
def RowShape (rows : TriangleRows) : Prop :=
  ∀ (i : Nat) (valid : i < rows.length), rows[i].2.length = i + 1
abbrev lengthTest : Expr :=
  .compare (.builtinCall .len [.name "row"]) [(.notEq, .name "size")]
abbrev shapeBody : List Stmt :=
  [.ifThen lengthTest [.return (.constant (.bool false))] [],
   .augAssign (.name "size") .add (.constant (.int 1))]

theorem rows_check (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (position : Nat) (env : HeapCalls.Env) (localSize : env "size" = some (.scalar (.int (position + 1)))) :
    ((∀ (i : Nat) (valid : i < rows.length), position ≤ i → rows[i].2.length = i + 1) ∧
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
    have savedSize : boundEnv "size" = some (.scalar (.int (position + 1))) := by
      simpa [boundEnv, HeapCalls.setLocal] using localSize
    have tested : HeapCalls.Evaluates program boundEnv lengthTest heap
        (.scalar (.bool (decide (pair.2.length ≠ position + 1)))) heap := by
      apply HeapCalls.Evaluates.compare (av := .scalar (.int pair.2.length)) (middle := heap)
      · exact HeapCalls.Evaluates.builtin (.cons (.lookup rfl) .nil)
          (by simp [HeapCalls.primitive, Mutable.length, reads pair (List.getElem_mem active), encode])
      · exact .lookup savedSize
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
    by_cases good : pair.2.length = position + 1
    · let nextEnv := HeapCalls.setLocal boundEnv "size" (.scalar (.int (position + 1 + 1)))
      have step : HeapCalls.BlockContinues program boundEnv shapeBody heap nextEnv heap := by
        have skip : HeapCalls.BlockContinues program boundEnv
            [.ifThen lengthTest [.return (.constant (.bool false))] []] heap boundEnv heap :=
          HeapCalls.BlockContinues.branch (flag := false) (by simpa [good] using tested) rfl HeapCalls.BlockContinues.nil
        apply skip.append
        exact HeapCalls.BlockContinues.augAssign savedSize HeapCalls.Evaluates.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
      rcases rows_check heap address rows found reads (position + 1) nextEnv rfl with
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

theorem is_triangle.result (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    (RowShape rows ∧
      HeapCalls.ReturnsValue program "is_triangle" [.ref address] heap (.scalar (.bool true)) heap) ∨
      HeapCalls.ReturnsValue program "is_triangle" [.ref address] heap (.scalar (.bool false)) heap := by
  let args : List Arg := [⟨"triangle", some (.subscript (.name "list")
    (.subscript (.name "list") (.name "int"))), none⟩]
  let env := HeapCalls.argumentEnv args [.ref address]
  let sized := HeapCalls.setLocal env "size" (.scalar (.int 1))
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by simp [Mutable.sequence, found]
  rcases rows_check heap address rows found reads 0 sized rfl with ⟨good, finalEnv, loop⟩ | rejected
  · left
    refine ⟨fun i valid => good i valid (by omega), ?_⟩
    apply HeapCalls.BlockReturns.function (args := args) rfl rfl
    apply HeapCalls.BlockReturns.assign (name := "size") HeapCalls.Evaluates.constant
    have entered : HeapCalls.BlockContinues program sized
        [.forIn (.name "row") (.name "triangle") shapeBody []] heap finalEnv heap :=
      HeapCalls.BlockContinues.for_sequence (by intros expressions impossible; cases impossible)
        (.lookup rfl) sequence loop
    exact entered.returns (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
  · right
    apply HeapCalls.BlockReturns.function (args := args) rfl rfl
    apply HeapCalls.BlockReturns.assign (name := "size") HeapCalls.Evaluates.constant
    exact HeapCalls.BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
      (.lookup rfl) sequence (rejected [.return (.constant (.bool true))])

def TriangleDomain (heap : Mutable.Heap) (address : Nat) : Prop :=
  ∃ rows : TriangleRows,
    heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
    (∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) ∧ RowShape rows

theorem decode_rows (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ rows : TriangleRows,
      heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
      ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)) := by
  classical
  obtain ⟨addresses, found, rowTypes⟩ := Mutable.Value.lists_of_hasType typed
  have decoded : ∀ row ∈ addresses, ∃ values : List Int, heap[row]? = some (.list (encode values)) :=
    fun row member => Mutable.Value.ints_of_hasType (rowTypes row member)
  choose values contents using decoded
  let rows := addresses.attach.map (fun entry => (entry.val, values entry.val entry.property))
  have same : rows.map (fun pair => Mutable.Value.ref pair.1) = addresses.map Mutable.Value.ref := by
    simp [rows, List.map_map]
  refine ⟨rows, by simpa [same] using found, ?_⟩
  intro pair member
  obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
  exact contents entry.val entry.property

abbrev triangleEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "triangle" then some (.ref address) else none

theorem triangle_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ flag, HeapCalls.Evaluates program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap
      (.scalar (.bool flag)) heap ∧ (flag = true → TriangleDomain heap address) := by
  obtain ⟨rows, found, reads⟩ := decode_rows heap address typed
  rcases is_triangle.result heap address rows found reads with ⟨shape, executed⟩ | executed
  · exact ⟨true, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) executed,
      fun _ => ⟨rows, found, reads, shape⟩⟩
  · exact ⟨false, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) executed, by simp⟩

theorem triangle_domain_of_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap) :
    TriangleDomain heap address := by
  obtain ⟨flag, evaluated, domain⟩ := triangle_condition heap address typed
  exact domain (by simpa using evaluated.condition_iff.mp requires)

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (triangleEnv address) Solvers.P67Ast.requires_solve_0 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := triangle_condition heap address typed
  exact Execution.supported_of_finishes evaluated.finishes


end solvers.«67»
