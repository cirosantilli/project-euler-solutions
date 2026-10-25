import «.formalsw».solvers.«52.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerSolutions.Lib.PermutedMultiples
import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.LoopCalls

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

namespace solve
private def code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private def widthBody : List Stmt := match (code[1]? : Option Stmt) with
  | some (.whileDo _ body _) => body | _ => []
private def candidateBody : List Stmt := match (widthBody[2]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def multipleBody : List Stmt := match (candidateBody[2]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def comparison : Expr := match (multipleBody[0]? : Option Stmt) with
  | some (.ifThen condition _ _) => condition | _ => .constant .none
private def signatureObjects (n : Nat) : Mutable.Heap :=
  [.list [number 0], .list (numbers (signature n)), .tuple (numbers (signature n))]

private theorem equal_numbers (left right : List Nat) :
    equalLists (numbers left) (numbers right) = some (decide (left = right)) := by
  have encoded : left.map (fun (n : Nat) => (n : Int)) = right.map (fun (n : Nat) => (n : Int)) ↔ left = right :=
    List.map_inj_right (fun _ _ equal => Int.natCast_inj.mp equal)
  simpa only [List.map_map, Function.comp_def, encoded] using
    equalLists_ints (left.map (fun (n : Nat) => (n : Int))) (right.map (fun (n : Nat) => (n : Int)))

/-- The actual tuple comparison reads the saved signature after the fresh helper allocation. -/
theorem compare_multiple (x m address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (number x)) (getM : env "m" = some (number m))
    (getSignature : env "sig" = some (.ref address)) (found : heap[address]? = some (.tuple (numbers (signature x)))) :
    Evaluates program env comparison heap (flag (!ProjectEulerSolutions.PermutedMultiples.same x (m*x)))
      (heap ++ signatureObjects (m*x)) := by
  let grown := heap ++ signatureObjects (m*x)
  have saved : grown[address]? = some (.tuple (numbers (signature x))) :=
    ProjectEulerSolutions.HeapCollection.prefix_found heap _ address _ found
  have fresh : grown[heap.length+2]? = some (.tuple (numbers (signature (m*x)))) := by simp [grown, signatureObjects]
  have distinct : heap.length+2 ≠ address := by
    have inside := (List.getElem?_eq_some_iff.mp found).1
    omega
  have returned := digit_signature.result (m*x : Nat) heap
  have normalized : ReturnsValue program "digit_signature" [number (m*x : Nat)] heap (.ref (heap.length+2)) grown := by
    simpa only [grown, signatureObjects, Int.toNat_natCast] using returned
  apply Evaluates.compare (av := .ref (heap.length+2)) (bv := .ref address) (middle := grown) (afterRight := grown)
  · apply Evaluates.call (middle := heap) (values := [number (m*x : Nat)])
    · refine .cons ?_ .nil
      exact Evaluates.binary (Evaluates.lookup getM) (Evaluates.lookup getX) (by simp [Mutable.binary, scalarBinary, number])
    · exact normalized
  · exact Evaluates.lookup getSignature
  · have swapped : decide (signature (m*x) = signature x) = ProjectEulerSolutions.PermutedMultiples.same x (m*x) := by
      simp [ProjectEulerSolutions.PermutedMultiples.same, eq_comm]
    simp [HeapCalls.compare, fresh, saved, distinct, equal_numbers, swapped, flag, grown, show (CmpOp.notEq == CmpOp.eq) = false from rfl]

private def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "m" → name ≠ "ok" → after name = before name

/-- The source multiple scan preserves captured locals and breaks at its first unequal signature. -/
theorem multiples_loop (x address first count stop : Nat) (length : first+count = stop)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getX : env "x" = some (number x)) (getSignature : env "sig" = some (.ref address))
    (getOk : env "ok" = some (flag true)) (found : heap[address]? = some (.tuple (numbers (signature x)))) :
    ∃ finalEnv extra, BlockContinues program env [.rangeStepLoop "m" first stop 1 multipleBody []]
      heap finalEnv (heap ++ extra) ∧ Stable env finalEnv ∧
      finalEnv "ok" = some (flag ((List.range' first count).all (fun m => ProjectEulerSolutions.PermutedMultiples.same x (m*x)))) := by
  induction count generalizing first env heap with
  | zero =>
      refine ⟨env, [], ?_, fun _ _ _ => rfl, ?_⟩
      · simpa only [List.append_nil] using BlockContinues.range_done
          (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
      · simpa using getOk
  | succ count ih =>
      let bound := setLocal env "m" (number first)
      have current := compare_multiple x first address heap bound (by simpa [bound, setLocal] using getX)
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getSignature) found
      have saved := ProjectEulerSolutions.HeapCollection.prefix_found heap (signatureObjects (first*x)) address _ found
      cases accepted : ProjectEulerSolutions.PermutedMultiples.same x (first*x) with
      | false =>
          let finalEnv := setLocal bound "ok" (flag false)
          refine ⟨finalEnv, signatureObjects (first*x), ?_, ?_, ?_⟩
          · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockContinues.branch_next (flag := true) (by simpa only [accepted, Bool.not_false] using current) rfl
            apply BlockContinues.append (body := [.assign [.name "ok"] (.constant (.bool false))])
              (middleEnv := finalEnv) (middleHeap := heap ++ signatureObjects (first*x))
              (BlockContinues.assign Evaluates.constant)
            exact BlockContinues.break_loop rfl BlockContinues.nil
          · intro name notM notOk
            simp [finalEnv, bound, setLocal, notM, notOk]
          · simp [List.range'_succ, accepted, finalEnv, setLocal]
      | true =>
          obtain ⟨finalEnv, extra, continued, stable, finalOk⟩ := ih (first+1) (by omega) bound
            (heap ++ signatureObjects (first*x)) (by simpa [bound, setLocal] using getX)
            (by simpa [bound, setLocal] using getSignature) (by simpa [bound, setLocal] using getOk) saved
          refine ⟨finalEnv, signatureObjects (first*x) ++ extra, ?_, ?_, ?_⟩
          · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockContinues.branch_next (flag := false) (by simpa only [accepted, Bool.not_true] using current) rfl
            have resumed := BlockContinues.loop_resume continued
            simpa only [List.append_assoc, multipleBody, Calls.loopContinuation, Calls.loopControl,
              Calls.loopControlStmt, Bool.false_or, Bool.true_or, if_true, List.nil_append, Nat.cast_add, Nat.cast_one] using resumed
          · intro name notM notOk
            exact (stable name notM notOk).trans (by simp [bound, setLocal, notM])
          · simpa only [List.range'_succ, List.all_cons, accepted, Bool.true_and] using finalOk
/-- The original per-candidate setup computes its signature and complete acceptance flag. -/
theorem prepare_candidate (limit x : Nat) (supported : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit_mul" = some (number limit)) (getX : env "x" = some (number x)) :
    ∃ finalEnv extra, BlockContinues program env (candidateBody.take 3) heap finalEnv (heap ++ extra) ∧
      (∀ name, name ≠ "sig" → name ≠ "ok" → name ≠ "m" → finalEnv name = env name) ∧
      finalEnv "ok" = some (flag (ProjectEulerSolutions.PermutedMultiples.accepts limit x)) := by
  let grown := heap ++ signatureObjects x
  let withSignature := setLocal env "sig" (.ref (heap.length+2))
  let ready := setLocal withSignature "ok" (flag true)
  obtain ⟨finalEnv, extra, scanned, stable, getOk⟩ := multiples_loop x (heap.length+2) 2 (limit-1) (limit+1)
    (by omega) ready grown (by simpa [ready, withSignature, setLocal] using getX)
    (by simp [ready, withSignature, setLocal]) (by simp [ready, setLocal]) (by simp [grown, signatureObjects])
  refine ⟨finalEnv, signatureObjects x ++ extra, ?_, ?_, ?_⟩
  · apply BlockContinues.append (body := [candidateBody[0]'(by decide)]) (middleEnv := withSignature) (middleHeap := grown)
    · apply BlockContinues.assign
      exact Evaluates.call (.cons (Evaluates.lookup getX) .nil)
        (by simpa only [grown, signatureObjects, Int.toNat_natCast] using digit_signature.result x heap)
    · apply BlockContinues.append (body := [candidateBody[1]'(by decide)]) (middleEnv := ready) (middleHeap := grown)
        (BlockContinues.assign Evaluates.constant)
      apply BlockContinues.for_range (middle := grown) (values := [number 2, number (limit+1 : Nat)])
        (start := 2) (stop := ((limit+1 : Nat) : Int)) (stride := 1)
      · refine .cons Evaluates.constant (.cons ?_ .nil)
        exact Evaluates.binary (Evaluates.lookup (by simpa [ready, withSignature, setLocal] using getLimit)) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary, number])
      · rfl
      · simpa only [grown, List.append_assoc] using scanned
  · intro name notSig notOk notM
    exact (stable name notM notOk).trans (by simp [ready, withSignature, setLocal, notSig, notOk])
  · have natural : ((limit : Int)-1).toNat = limit-1 := by omega
    simpa only [ProjectEulerSolutions.PermutedMultiples.accepts, show decide (2 ≤ (limit : Int)) = true from by simp; omega,
      Bool.true_and, natural, List.range'_eq_map_range, List.all_map, Nat.add_comm] using getOk

end solve

end solvers.«52»
