import «.formalsw».solvers.«52.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DigitHistogram

/-! Source proofs for problem 52's decimal digit counters and permuted-multiple search. -/
set_option Elab.async false

namespace solvers.«52»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.DigitHistogram
private abbrev program := Solvers.P52Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def numbers (items : List Nat) : List Mutable.Value := items.map (fun (n : Nat) => number n)

namespace digit_signature
private def code : List Stmt := executableBody ((findFunctionIn program.body "digit_signature").getD ([], [])).2
private def body : List Stmt := match (code[1]? : Option Stmt) with
  | some (.whileDo _ statements _) => statements | _ => []
private def table (digits : List Nat) : List Mutable.Value := numbers (counts digits)
private def working (base : Mutable.Heap) (digits : List Nat) : Mutable.Heap := base ++ [.list (table digits)]

private theorem increment (n : Nat) (acc : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getCounts : env "cnt" = some (.ref base.length)) :
    BlockContinues program env [body[0]'(by decide)] (working base acc) env (working base (acc ++ [n%10])) := by
  have within : n%10 < 10 := Nat.mod_lt _ (by decide)
  have item : (table acc)[n%10]? = some (number (acc.count (n%10))) := by
    simp [table, numbers, within]
  have changed : (working base acc).set base.length
      (.list ((table acc).set (n%10) (number (acc.count (n%10)+1)))) = working base (acc ++ [n%10]) := by
    simp only [working, table, numbers, append_digit, List.map_set]
    simp
  rw [← changed]
  apply BlockContinues.augItem (previous := number (acc.count (n%10))) (right := number 1)
    (updated := number (acc.count (n%10)+1)) (key := number (n%10 : Nat))
    (afterReceiver := working base acc) (afterIndex := working base acc) (afterRead := working base acc)
    (afterRhs := working base acc) (afterOperation := working base acc)
    (by intro a b c impossible; cases impossible) (Evaluates.lookup getCounts)
  · exact Evaluates.binary (Evaluates.lookup getN) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
  · have read := Mutable.getItem_list (working base acc) base.length (n%10) (table acc)
      (by simp [working]) (by simpa [table, numbers] using within)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using read
  · exact Evaluates.constant
  · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
  · have nonnegative : ¬ (n : Int)%10 < 0 := by omega
    have inside : ¬ 10 ≤ (n : Int)%10 := by omega
    have index : ((n : Int)%10).toNat = n%10 := by omega
    simp [Mutable.setItem, working, Mutable.index, table, numbers, nonnegative, inside, index, Except.map, number]

private theorem finish (base : Mutable.Heap) (acc : List Nat) (env : HeapCalls.Env)
    (getCounts : env "cnt" = some (.ref base.length)) :
    BlockReturns program env (code.drop 2) (working base acc) (.ref (base.length+1))
      (working base acc ++ [.tuple (table acc)]) := by
  apply BlockReturns.return_value
  apply Evaluates.builtin (.cons (Evaluates.lookup getCounts) .nil)
  have converted := primitive_tuple_list (working base acc) base.length (table acc) (by simp [working])
  simpa [Mutable.allocateTuple, working, table, numbers, counts, Except.map] using converted

private theorem loop (n : Int) (acc : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getCounts : env "cnt" = some (.ref base.length)) :
    BlockReturns program env (code.drop 1) (working base acc) (.ref (base.length+1))
      (working base (acc ++ Nat.digits 10 n.toNat) ++ [.tuple (table (acc ++ Nat.digits 10 n.toNat))]) := by
  by_cases positive : 0 < n
  · lift n to Nat using (by omega : 0 ≤ n)
    have next := loop (n/10 : Nat) (acc ++ [n%10]) base (setLocal env "n" (number (n/10 : Nat)))
      (by simp [setLocal]) (by simpa [setLocal] using getCounts)
    have normalized : (acc ++ [n%10]) ++ Nat.digits 10 (n/10) = acc ++ Nat.digits 10 n := by
      rw [Nat.digits_eq_cons_digits_div (by decide : 1 < 10) (by omega : n ≠ 0)]
      simp
    simp only [Int.toNat_natCast] at next ⊢
    rw [normalized] at next
    apply BlockReturns.while_loop (flag := true) (value := flag true) (middle := working base acc)
      (Evaluates.compare (av := number n) (bv := number 0) (middle := working base acc) (afterRight := working base acc)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, show 0 < n by omega])) rfl
    apply (increment n acc base env getN getCounts).returns
    apply BlockReturns.augAssign getN Evaluates.constant
      (value := number (n/10 : Nat)) (afterOperation := working base (acc ++ [n%10]))
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · exact next
  · have zero : n.toNat = 0 := by omega
    simp only [zero, Nat.digits_zero, List.append_nil]
    apply BlockReturns.while_loop (flag := false) (value := flag false) (middle := working base acc)
      (Evaluates.compare (av := number n) (bv := number 0) (middle := working base acc) (afterRight := working base acc)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, positive])) rfl
    exact finish base acc env getCounts
termination_by n.toNat
decreasing_by
  simp only [Int.toNat_natCast]
  have smaller := Nat.div_lt_self (by omega : 0 < n) (by decide : 1 < 10)
  omega

/-- The source computes all ten counters, retaining its scratch list and returning a fresh tuple. -/
theorem result (n : Int) (heap : Mutable.Heap) :
    ReturnsValue program "digit_signature" [number n] heap (.ref (heap.length+2))
      (heap ++ [.list [number 0], .list (numbers (signature n.toNat)), .tuple (numbers (signature n.toNat))]) := by
  let base := heap ++ [Mutable.Object.list [number 0]]
  let entry := functionEnv ((findFunctionIn program.body "digit_signature").getD ([], [])).1
    ((findFunctionIn program.body "digit_signature").getD ([], [])).2 [number n] []
  let ready := setLocal entry "cnt" (.ref base.length)
  have finished := loop n [] base ready rfl rfl
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref base.length) (middle := working base [])
  · apply Evaluates.binary (av := .ref heap.length) (bv := number 10)
      (middle := base) (afterRight := base)
      (Evaluates.list (.cons Evaluates.constant .nil)) Evaluates.constant
    simp [Mutable.binary, Mutable.array, Mutable.allocate, base, working, table, numbers, number, counts, Except.map, show (BinOp.mult == BinOp.mult) = true from rfl]
  · simpa [ready, entry, base, working, table, signature, List.append_assoc, Nat.add_assoc] using finished

theorem «Returns» (n : Int) : (HeapCalls.call program "digit_signature" [number n] []).Returns := (result n []).returns

theorem «Supported» (n : Int) : (HeapCalls.call program "digit_signature" [number n] []).Supported := («Returns» n).supported
end digit_signature

end solvers.«52»
