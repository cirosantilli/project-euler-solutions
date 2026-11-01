import «.formalsw».solvers.«88.py.ast»
import ProjectEulerSolutions.Lib.ProductSumNumbers
import lang.py.FormalSW.HeapCalls

namespace solvers.«88»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

abbrev program := Solvers.P88Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev dfsCode : List Stmt := executableBody
  ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
def dfsBody : List Stmt := match (dfsCode[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def measuredEnv (env : HeapCalls.Env) (before : List Nat) (factor : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "new_prod" (number (before.prod * factor)))
    "new_sum" (number (before.sum + factor))) "new_terms" (number (before.length + 1)))
    "k" (number (effectiveSize (before ++ [factor])))

/-- The four source assignments compute the next factor state without mutating the heap. -/
theorem measure_factor (before : List Nat) (factor : Nat)
    (valid : Admissible before) (factorBound : 2 ≤ factor)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getProduct : env "prod" = some (number before.prod))
    (getSum : env "summ" = some (number before.sum))
    (getTerms : env "terms" = some (number before.length))
    (getFactor : env "f" = some (number factor)) :
    BlockContinues program env (dfsBody.take 4) heap (measuredEnv env before factor) heap := by
  let e1 := setLocal env "new_prod" (number (before.prod * factor))
  let e2 := setLocal e1 "new_sum" (number (before.sum + factor))
  let e3 := setLocal e2 "new_terms" (number (before.length + 1))
  have first : BlockContinues program env [dfsBody[0]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getProduct) (Evaluates.lookup getFactor)
      (by simp [Mutable.binary, scalarBinary, number])
  have second : BlockContinues program e1 [dfsBody[1]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number before.sum) (bv := number factor)
      (Evaluates.lookup (by simpa [e1, setLocal] using getSum))
      (Evaluates.lookup (by simpa [e1, setLocal] using getFactor))
      (by simp [Mutable.binary, scalarBinary, number])
  have third : BlockContinues program e2 [dfsBody[2]'(by decide)] heap e3 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number before.length) (bv := number 1)
      (Evaluates.lookup (by simpa [e2, e1, setLocal] using getTerms)) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])
  have fourth : BlockContinues program e3 [dfsBody[3]'(by decide)] heap
      (measuredEnv env before factor) heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (before.length + 1)) (afterRight := heap)
      (bv := integer (((before.prod * factor : Nat) : Int) - (before.sum + factor)))
      (Evaluates.lookup (by simp [e3, setLocal]))
    · exact Evaluates.binary (av := number (before.prod * factor)) (bv := number (before.sum + factor))
        (Evaluates.lookup (by simp [e3, e2, e1, setLocal]))
        (Evaluates.lookup (by simp [e3, e2, setLocal]))
        (by simp [Mutable.binary, scalarBinary, integer])
    · have single : Admissible [factor] := by simpa [Admissible] using factorBound
      have formula := effectiveSize_int (before ++ [factor])
        (admissible_append before [factor] valid single)
      simp only [List.length_append, List.length_cons, List.length_nil, List.prod_append,
        List.prod_cons, List.prod_nil, mul_one, List.sum_append, List.sum_cons,
        List.sum_nil, add_zero, Nat.cast_add, Nat.cast_mul, Nat.cast_one, Nat.cast_zero] at formula
      simp [Mutable.binary, scalarBinary, number, formula, sub_eq_add_neg, add_assoc]
  exact first.append (second.append (third.append fourth))

theorem measured_lookup (env : HeapCalls.Env) (before : List Nat) (factor : Nat) :
    measuredEnv env before factor "k" = some (number (effectiveSize (before ++ [factor]))) := by
  simp [measuredEnv, setLocal]

theorem measured_preserves (env : HeapCalls.Env) (before : List Nat) (factor : Nat) (name : String)
    (fresh : name ≠ "new_prod" ∧ name ≠ "new_sum" ∧ name ≠ "new_terms" ∧ name ≠ "k") :
    measuredEnv env before factor name = env name := by
  simp [measuredEnv, setLocal, fresh.1, fresh.2.1, fresh.2.2.1, fresh.2.2.2]

end solvers.«88»
