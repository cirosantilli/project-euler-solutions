import solvers.«34.table.source»

/-! Source execution of digit-factorial summation in every supported base. -/
namespace solvers.«34»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.DigitFactorialBase
set_option Elab.async false
private abbrev program := Solvers.P34Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem read_factorial (base : Nat) (heap : Mutable.Heap) (address d : Nat) (inside : d < base)
    (found : heap[address]? = some (.list (factorialTable base))) (env : HeapCalls.Env)
    (getFacts : env "facts" = some (.ref address))
    (getRemainder : env "remainder" = some (number d)) :
    Evaluates program env (.subscript (.name "facts") (.name "remainder")) heap
      (number (Nat.factorial d)) heap := by
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getFacts) (Evaluates.lookup getRemainder)
  have entry : (factorialTable base)[d]? = some (number (Nat.factorial d)) := by simp [factorialTable, facts, inside]
  simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
    Mutable.getItem_list heap address d (factorialTable base) found (by simpa [factorialTable] using inside)


private abbrev digitBody : List Stmt :=
  [.assign [.tuple [.name "n", .name "remainder"]] (.builtinCall .divmod [.name "n", .name "base"]),
   .augAssign (.name "total") .add (.subscript (.name "facts") (.name "remainder"))]
private abbrev digitWhile : Stmt :=
  .whileDo (.name "n") digitBody []

/-- Divmod creates temporary tuples, but never changes the input factorial table. -/
private theorem digit_loop (base n total address : Nat) (valid : 2 ≤ base) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (factorialTable base)))
    (getN : env "n" = some (number n)) (getS : env "total" = some (number total))
    (getFacts : env "facts" = some (.ref address)) (getBase : env "base" = some (number base)) :
    ∃ extra, BlockReturns program env [digitWhile, .return (.name "total")] heap
      (number (total + digitSum base n)) (heap ++ extra) := by
  by_cases positive : 0 < n
  · let pair : Mutable.Object := .tuple [number (n / base), number (n % base)]
    let unpacked := assignNames env ["n", "remainder"] [number (n / base), number (n % base)]
    let updated := setLocal unpacked "total" (number (total + Nat.factorial (n % base)))
    obtain ⟨extra, rest⟩ := digit_loop base (n / base) (total + Nat.factorial (n % base)) address valid
      (heap ++ [pair]) updated (found_append found [pair])
      (by simp [updated, unpacked, assignNames, setLocal]) (by simp [updated, setLocal])
      (by simpa [updated, unpacked, assignNames, setLocal] using getFacts)
      (by simpa [updated, unpacked, assignNames, setLocal] using getBase)
    refine ⟨[pair] ++ extra, ?_⟩
    rw [← List.append_assoc]
    apply BlockReturns.while_loop (flag := true)
      (Evaluates.lookup getN)
      (by simp [Mutable.truth, scalarTruth]; omega)
    apply BlockReturns.unpack (items := [number (n / base), number (n % base)])
      (afterUnpack := heap ++ [pair]) rfl
    · apply Evaluates.builtin (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getBase) .nil))
      have divided := Mutable.divmod_ints (n : Int) (base : Int) (by omega) heap
      have quotient : (n : Int).fdiv (base : Int) = ((n / base : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]; rfl
      have remainder : (n : Int) - (base : Int) * (n : Int).fdiv (base : Int) = ((n % base : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), ← Int.emod_def]; rfl
      simpa only [quotient, remainder] using divided
    · simp [Mutable.sequence, pair, Int.emod_def]
    · rfl
    · apply BlockReturns.augAssign (left := number total) (right := number (Nat.factorial (n % base)))
        (value := number (total + Nat.factorial (n % base))) (middle := heap ++ [pair])
        (afterOperation := heap ++ [pair])
        (by simpa [assignNames, setLocal] using getS)
      · exact read_factorial base _ address (n % base) (Nat.mod_lt _ (by omega))
          (found_append found [pair]) _
          (by simpa [assignNames, setLocal] using getFacts) (by simp [assignNames, setLocal])
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa [digitWhile, digitBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          updated, unpacked, digitSum_step base n valid positive, Nat.add_assoc] using rest
  · have zero : n = 0 := by omega
    subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false)
      (Evaluates.lookup getN) rfl
    simpa [digitSum_zero] using BlockReturns.return_value (module := program) (Evaluates.lookup getS)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by omega)

theorem digit_factorial_sum.result (base n address : Nat) (valid : 2 ≤ base) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (factorialTable base))) :
    ∃ extra, ReturnsValue program "digit_factorial_sum" [number n, .ref address, number base] heap
      (number (digitSum base n)) (heap ++ extra) := by
  obtain ⟨extra, loop⟩ := digit_loop base n 0 address valid heap
    (setLocal (functionEnv ((findFunctionIn program.body "digit_factorial_sum").getD ([], [])).1
      ((findFunctionIn program.body "digit_factorial_sum").getD ([], [])).2 [number n, .ref address, number base] [])
      "total" (number 0)) found rfl rfl rfl (by simp [setLocal]; rfl)
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  simpa only [Nat.zero_add] using loop

end solvers.«34»
