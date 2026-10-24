import solvers.«37.prime.source»
import ProjectEulerStatements.Lib.TruncatablePrimeDigits

set_option Elab.async false

namespace solvers.«37».is_left_truncatable
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerStatements.TruncatablePrimeBase
private abbrev program := Solvers.P37Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

private def remaining (radix n first : Nat) : Bool :=
  (List.range' first ((Nat.digits radix n).length - first)).all
    fun count => decide (Nat.Prime (n % radix ^ count))

private theorem remaining_step (radix n first : Nat)
    (inside : first < (Nat.digits radix n).length) :
    remaining radix n first =
      (decide (Nat.Prime (n % radix ^ first)) && remaining radix n (first + 1)) := by
  unfold remaining
  rw [show (Nat.digits radix n).length - first =
    ((Nat.digits radix n).length - (first + 1)) + 1 by omega]
  simp only [List.range'_succ, List.all_cons]

private abbrev body : List Stmt :=
  [.ifThen (.unaryOp .not (.call (.name "is_prime")
      [.binOp (.name "n") .mod (.name "power")])) [.return (.constant (.bool false))] [],
   .augAssign (.name "power") .mult (.name "base")]
private abbrev condition : Expr := .compare (.name "power") [(.ltE, .name "n")]
private abbrev searchLoop : Stmt := .whileDo condition body []

private theorem loop (radix n first : Nat) (valid : 2 ≤ radix)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getBase : env "base" = some (number radix))
    (getPower : env "power" = some (number (radix ^ first))) :
    BlockReturns program env [searchLoop, .return (.constant (.bool true))]
      heap (flag (remaining radix n first)) heap := by
  induction measure : (Nat.digits radix n).length - first using Nat.strong_induction_on
      generalizing first env with
  | h size ih =>
      have compared : Evaluates program env condition heap
          (flag (decide (radix ^ first ≤ n))) heap := by
        apply Evaluates.compare (av := number (radix ^ first)) (bv := number n)
          (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.lookup getPower) (Evaluates.lookup getN)
        change some (Except.ok (ε := ExceptionObject)
          (flag (decide (((radix ^ first : Nat) : Int) ≤ (n : Int)))), heap) = _
        simp only [Int.ofNat_le]
      by_cases inside : radix ^ first ≤ n
      · have width := (Nat.lt_digits_length_iff valid n).mpr inside
        rw [remaining_step radix n first width]
        apply BlockReturns.while_loop (middle := heap) (flag := true) (value := flag true)
          (by simpa only [inside, decide_true] using compared) rfl
        have checked : Evaluates program env
            (.unaryOp .not (.call (.name "is_prime") [.binOp (.name "n") .mod (.name "power")]))
            heap (flag (!decide (Nat.Prime (n % radix ^ first)))) heap := by
          apply Evaluates.unary (middle := heap) (value := flag (decide (Nat.Prime (n % radix ^ first))))
          · apply Evaluates.call
              (.cons (Evaluates.binary (av := number n) (bv := number (radix ^ first))
                (value := number (n % radix ^ first))
                (middle := heap) (afterRight := heap) (finalHeap := heap)
                (Evaluates.lookup getN) (Evaluates.lookup getPower)
                (by simp [Mutable.binary, scalarBinary, number, show radix ≠ 0 by omega,
                  show 0 < (radix : Int) ^ first from pow_pos (by omega) _])) .nil)
            exact solvers.«37».is_prime.result (n % radix ^ first) heap
          · simp [HeapCalls.unary, Mutable.truth, flag, scalarTruth]
        by_cases prime : Nat.Prime (n % radix ^ first)
        · simp only [prime, decide_true, Bool.true_and]
          apply BlockReturns.branch (middle := heap) (flag := false) (value := flag false)
            (by simpa only [prime, decide_true, Bool.not_true] using checked) rfl
          apply BlockReturns.augAssign (left := number (radix ^ first)) (right := number radix)
            (value := number (radix ^ (first + 1))) (afterOperation := heap)
            getPower (Evaluates.lookup getBase)
          · simp [inplace, Mutable.binary, scalarBinary, number, Nat.pow_succ]
          · have rest := ih ((Nat.digits radix n).length - (first + 1)) (by omega)
              (first + 1) (setLocal env "power" (number (radix ^ (first + 1))))
              (by simpa [setLocal] using getN) (by simpa [setLocal] using getBase)
              (by simp [setLocal]) rfl
            simpa [searchLoop, body, Calls.loopContinuation, Calls.loopControl,
              Calls.loopControlStmt] using rest
        · simp only [prime, decide_false, Bool.false_and]
          apply BlockReturns.branch (middle := heap) (flag := true) (value := flag true)
            (by simpa only [prime, decide_false, Bool.not_false] using checked) rfl
          exact BlockReturns.return_value Evaluates.constant
      · have outside : (Nat.digits radix n).length ≤ first := by
          by_contra outside
          exact inside ((Nat.lt_digits_length_iff valid n).mp (by omega))
        simp only [remaining, Nat.sub_eq_zero_of_le outside, List.range'_zero, List.all_nil]
        apply BlockReturns.while_loop (middle := heap) (flag := false) (value := flag false)
          (by simpa only [inside, decide_false] using compared) rfl
        exact BlockReturns.return_value Evaluates.constant

theorem result (radix n : Nat) (valid : 2 ≤ radix) (heap : Mutable.Heap) :
    ReturnsValue program "is_left_truncatable" [number n, number radix] heap
      (flag (leftCheck radix n)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.lookup rfl)
  simpa only [leftCheck, remaining] using loop radix n 1 valid heap _ rfl rfl
    (by simp [setLocal])

end solvers.«37».is_left_truncatable
