import solvers.«74.py.lean.d».Factorials
import ProjectEulerStatements.Lib.DigitFactorialChains

/-! The optimized digit loop reads its initialized global table and preserves old heap objects. -/
namespace solvers.«74».next_term
open FormalSW.PyAst HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P74Ast.module
private abbrev number := factorials.number

private def rawSum (n : Nat) : Nat := ((Nat.digits 10 n).map Nat.factorial).sum

private theorem rawSum_step (n : Nat) (positive : 0 < n) :
    rawSum n = Nat.factorial (n % 10) + rawSum (n / 10) := by
  simp only [rawSum, Nat.digits_def' (by decide : 1 < 10) positive,
    List.map_cons, List.sum_cons]

private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem read_factorial (digit address : Nat) (small : digit < 10)
    (heap : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list factorials.values))
    (getTable : locals "FACTORIALS" = some (.ref address)) (index : Expr)
    (ordinary : ∀ a b c, index ≠ .slice a b c)
    (evaluated : Evaluates program locals index heap (number digit) heap globals) :
    Evaluates program locals (.subscript (.name "FACTORIALS") index) heap
      (number (Nat.factorial digit)) heap globals := by
  apply Evaluates.subscript ordinary (Evaluates.lookup getTable) evaluated
  have entry := factorials.value_at digit small
  simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
    Mutable.getItem_list heap address digit factorials.values found
      (by simpa [factorials.values] using small)

private abbrev digitBody : List Stmt :=
  [.assign [.tuple [.name "n", .name "digit"]]
      (.builtinCall .divmod [.name "n", .constant (.int 10)]),
    .augAssign (.name "total") .add (.subscript (.name "FACTORIALS") (.name "digit"))]
private abbrev digitWhile : Stmt := .whileDo (.name "n") digitBody []

private theorem digit_loop (n total address : Nat) (heap : Mutable.Heap)
    (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list factorials.values))
    (getN : locals "n" = some (number n)) (getTotal : locals "total" = some (number total))
    (getTable : locals "FACTORIALS" = some (.ref address)) :
    ∃ extra, BlockReturns program locals [digitWhile, .return (.name "total")] heap
      (number (total + rawSum n)) (heap ++ extra) globals := by
  by_cases positive : 0 < n
  · let pair : Mutable.Object := .tuple [number (n / 10), number (n % 10)]
    let unpacked := assignNames locals ["n", "digit"] [number (n / 10), number (n % 10)]
    let updated := setLocal unpacked "total" (number (total + Nat.factorial (n % 10)))
    obtain ⟨extra, rest⟩ := digit_loop (n / 10) (total + Nat.factorial (n % 10)) address
      (heap ++ [pair]) updated globals (found_append found [pair])
      (by simp [updated, unpacked, assignNames, setLocal]) (by simp [updated, setLocal])
      (by simpa [updated, unpacked, assignNames, setLocal] using getTable)
    refine ⟨[pair] ++ extra, ?_⟩
    rw [← List.append_assoc]
    apply BlockReturns.while_loop (flag := true) (Evaluates.lookup getN)
      (by simp [Mutable.truth, scalarTruth]; omega)
    apply BlockReturns.unpack (items := [number (n / 10), number (n % 10)])
      (afterUnpack := heap ++ [pair]) rfl
    · apply Evaluates.builtin (.cons (Evaluates.lookup getN) (.cons Evaluates.constant .nil))
      have divided := Mutable.divmod_ints (n : Int) 10 (by decide) heap
      have quotient : (n : Int).fdiv 10 = ((n / 10 : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
      have remainder : (n : Int) - 10 * (n : Int).fdiv 10 = ((n % 10 : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
      simpa only [quotient, remainder] using divided
    · simp [Mutable.sequence, pair, Int.emod_def]
    · rfl
    · apply BlockReturns.augAssign (left := number total)
        (right := number (Nat.factorial (n % 10)))
        (value := number (total + Nat.factorial (n % 10)))
        (middle := heap ++ [pair]) (afterOperation := heap ++ [pair])
        (by simpa [assignNames, setLocal] using getTotal)
      · exact read_factorial (n % 10) address (Nat.mod_lt _ (by decide)) _ _ globals
          (found_append found [pair])
          (by simpa [assignNames, setLocal] using getTable) _
          (by intro a b c; simp) (Evaluates.lookup (by simp [assignNames, setLocal]))
      · simp [inplace, Mutable.binary, scalarBinary, number, factorials.number]
      · simpa [digitWhile, digitBody, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, updated, unpacked, rawSum_step n positive, Nat.add_assoc]
          using rest
  · have zero : n = 0 := by omega
    subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false) (Evaluates.lookup getN) rfl
    simpa [rawSum] using
      BlockReturns.return_value (module := program) (Evaluates.lookup getTotal)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by decide)

/-- The table requirement is discharged by the actual module initialization theorem. -/
theorem result (n address : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getTable : globals.lookup "FACTORIALS" = some (.ref address))
    (found : heap[address]? = some (.list factorials.values)) :
    ∃ extra, ReturnsValue program "next_term" [number n] heap
      (number (ProjectEulerStatements.DigitFactorialChains.nextTerm n))
      (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "next_term").getD ([], [])).1
  let source := ((findFunctionIn program.body "next_term").getD ([], [])).2
  let entry := functionEnv args source [number n] globals
  have input : entry "n" = some (number n) := by cases globals <;> rfl
  have argsShape : args = [⟨"n", some (.name "int"), none⟩] := rfl
  have sourceShape : source =
      [.ifThen (.compare (.name "n") [(.eq, .constant (.int 0))])
        [.return (.subscript (.name "FACTORIALS") (.constant (.int 0)))] [],
       .assign [.name "total"] (.constant (.int 0)), digitWhile, .return (.name "total")] := rfl
  have localNames : blockLocals source = ["total", "n", "digit", "total"] := by
    rw [sourceShape]
    simp [blockLocals, assignedLocals, targetLocals]
  have table : entry "FACTORIALS" = some (.ref address) := by
    have captured : entry "FACTORIALS" = globals.lookup "FACTORIALS" := by
      cases globals <;> simp [entry, functionEnv, localNames, argsShape, HeapCalls.argumentEnv]
    exact captured.trans getTable
  by_cases zero : n = 0
  · subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil, ProjectEulerStatements.DigitFactorialChains.nextTerm_zero]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
      (Evaluates.compare (Evaluates.lookup input) Evaluates.constant rfl) rfl
    apply BlockReturns.return_value
    simpa only [Nat.factorial_zero] using read_factorial 0 address (by decide) heap entry
      globals found table (.constant (.int 0)) (by intro a b c; simp) Evaluates.constant
  · let ready := setLocal entry "total" (number 0)
    obtain ⟨extra, executed⟩ := digit_loop n 0 address heap ready globals found
      (by simpa [ready, setLocal] using input) (by simp [ready, setLocal])
      (by simpa [ready, setLocal] using table)
    refine ⟨extra, ?_⟩
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
      (Evaluates.compare (Evaluates.lookup input) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
    apply BlockReturns.assign Evaluates.constant
    simpa only [Nat.zero_add, rawSum, ProjectEulerStatements.DigitFactorialChains.nextTerm,
      if_neg zero] using executed

theorem initialized_result (n : Nat) (caller : Mutable.Heap) :
    ∃ extra, ReturnsValue program "next_term" [number n] (factorials.heap caller)
      (number (ProjectEulerStatements.DigitFactorialChains.nextTerm n))
      (factorials.heap caller ++ extra) (factorials.globals caller) :=
  result n caller.length (factorials.heap caller) (factorials.globals caller)
    (by simp [factorials.globals]) (factorials.found caller)

end solvers.«74».next_term
