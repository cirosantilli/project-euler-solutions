import «.formalsw».solvers.«92.py.ast»
import ProjectEulerSolutions.Lib.DigitSquareChains
import lang.py.FormalSW.HeapCalls

namespace solvers.«92».next_term
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

private abbrev digitBody : List Stmt :=
  [.assign [.tuple [.name "n", .name "digit"]]
      (.builtinCall .divmod [.name "n", .constant (.int 10)]),
    .augAssign (.name "total") .add (.binOp (.name "digit") .mult (.name "digit"))]
private abbrev digitWhile : Stmt := .whileDo (.name "n") digitBody []

private theorem digit_loop (n total : Nat) (heap : Mutable.Heap)
    (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (getN : locals "n" = some (number n)) (getTotal : locals "total" = some (number total)) :
    ∃ extra, BlockReturns program locals [digitWhile, .return (.name "total")] heap
      (number (total + nextTerm n)) (heap ++ extra) globals := by
  by_cases positive : 0 < n
  · let pair : Mutable.Object := .tuple [number (n / 10), number (n % 10)]
    let unpacked := assignNames locals ["n", "digit"] [number (n / 10), number (n % 10)]
    let updated := setLocal unpacked "total" (number (total + (n % 10) * (n % 10)))
    obtain ⟨extra, rest⟩ := digit_loop (n / 10) (total + (n % 10) * (n % 10))
      (heap ++ [pair]) updated globals
      (by simp [updated, unpacked, assignNames, setLocal]) (by simp [updated, setLocal])
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
        (right := number ((n % 10) * (n % 10)))
        (value := number (total + (n % 10) * (n % 10)))
        (middle := heap ++ [pair]) (afterOperation := heap ++ [pair])
        (by simpa [assignNames, setLocal] using getTotal)
      · exact Evaluates.binary (av := number (n % 10)) (bv := number (n % 10))
          (Evaluates.lookup (by simp [assignNames, setLocal]))
          (Evaluates.lookup (by simp [assignNames, setLocal]))
          (by simp [Mutable.binary, scalarBinary, number])
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa [digitWhile, digitBody, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, updated, unpacked, nextTerm_step n, Nat.add_assoc]
          using rest
  · have zero : n = 0 := by omega
    subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false) (Evaluates.lookup getN) rfl
    simpa using BlockReturns.return_value (module := program) (Evaluates.lookup getTotal)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by decide)

/-- Every quotient/remainder tuple is fresh; all objects in the caller's heap are retained. -/
theorem result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "next_term" [number n] heap
      (number (nextTerm n)) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "next_term").getD ([], [])).1
  let source := ((findFunctionIn program.body "next_term").getD ([], [])).2
  let entry := functionEnv args source [number n] globals
  have input : entry "n" = some (number n) := by cases globals <;> rfl
  let ready := setLocal entry "total" (number 0)
  obtain ⟨extra, executed⟩ := digit_loop n 0 heap ready globals
    (by simpa [ready, setLocal] using input) (by simp [ready, setLocal])
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  simpa only [Nat.zero_add] using executed

end solvers.«92».next_term
