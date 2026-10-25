import «.formalsw».solvers.«52.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DigitHistogram
import ProjectEulerSolutions.Lib.PermutedMultiples
import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.LoopCalls
import data.«project-euler-statements».data.py.project_euler_statements.«p52.py»

/-! Source-level termination and equivalence proofs for problem 52's complete permuted-multiple search. -/
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

private def SearchStable (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ ["x", "sig", "ok", "m"] → after name = before name

private theorem search_stable_trans (first : SearchStable before middle) (second : SearchStable middle after) :
    SearchStable before after := fun name untouched => (second name untouched).trans (first name untouched)

private theorem search_stable_bind (env : HeapCalls.Env) (x : Nat) : SearchStable env (setLocal env "x" (number x)) := by
  intro name untouched
  have different : name ≠ "x" := by intro same; exact untouched (by simp [same])
  simp [setLocal, different]

private def Search (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap) : Option Nat → Prop
  | some value => ∃ extra, ∀ following, BlockReturns program env (body ++ following) heap (number value) (heap ++ extra)
  | none => ∃ finalEnv extra, BlockContinues program env body heap finalEnv (heap ++ extra) ∧ SearchStable env finalEnv

private theorem candidate (limit x : Nat) (supported : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit_mul" = some (number limit)) (getX : env "x" = some (number x)) :
    Search env candidateBody heap (if ProjectEulerSolutions.PermutedMultiples.accepts limit x then some x else none) := by
  obtain ⟨ready, allocated, prepared, stable, getOk⟩ := prepare_candidate limit x supported env heap getLimit getX
  have current : ready "x" = some (number x) := (stable "x" (by decide) (by decide) (by decide)).trans getX
  cases accepted : ProjectEulerSolutions.PermutedMultiples.accepts limit x with
  | true =>
      simp only [↓reduceIte]
      refine ⟨allocated, ?_⟩
      intro following
      have tail : BlockReturns program ready (candidateBody.drop 3 ++ following) (heap ++ allocated) (number x) (heap ++ allocated) := by
        apply BlockReturns.branch (flag := true) (Evaluates.lookup (by simpa only [accepted] using getOk)) rfl
        exact BlockReturns.return_value (Evaluates.lookup current)
      simpa only [List.take_append_drop, List.append_assoc] using prepared.returns tail
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      refine ⟨ready, allocated, ?_, ?_⟩
      · have tail : BlockContinues program ready (candidateBody.drop 3) (heap ++ allocated) ready (heap ++ allocated) :=
          BlockContinues.branch (flag := false) (Evaluates.lookup (by simpa only [accepted] using getOk)) rfl .nil
        simpa only [List.take_append_drop] using prepared.append tail
      · intro name untouched
        have notSig : name ≠ "sig" := by intro same; exact untouched (by simp [same])
        have notOk : name ≠ "ok" := by intro same; exact untouched (by simp [same])
        have notM : name ≠ "m" := by intro same; exact untouched (by simp [same])
        exact stable name notSig notOk notM

/-- The complete source range returns its first accepted number, preserving captured width bounds on exhaustion. -/
theorem candidates_loop (limit first count stop : Nat) (supported : 2 ≤ limit) (length : first+count = stop)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getLimit : env "limit_mul" = some (number limit)) :
    Search env [.rangeStepLoop "x" first stop 1 candidateBody []] heap
      ((List.range' first count).find? (fun x => ProjectEulerSolutions.PermutedMultiples.accepts limit x)) := by
  induction count generalizing first env heap with
  | zero =>
      refine ⟨env, [], ?_, fun _ _ => rfl⟩
      simpa only [List.append_nil] using BlockContinues.range_done
        (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := setLocal env "x" (number first)
      have tested := candidate limit first supported bound heap (by simpa [bound, setLocal] using getLimit) (by simp [bound, setLocal])
      have active : Calls.rangeContinues first stop 1 = true := by simp [Calls.rangeContinues]; omega
      cases accepted : ProjectEulerSolutions.PermutedMultiples.accepts limit first with
      | true =>
          simp only [accepted, ↓reduceIte] at tested
          obtain ⟨allocated, returned⟩ := tested
          simp only [List.range'_succ, List.find?_cons, accepted]
          refine ⟨allocated, ?_⟩
          intro following
          apply BlockReturns.range_step active
          exact returned (.rangeStepLoop "x" ((first : Int)+1) stop 1 candidateBody [] :: following)
      | false =>
          simp only [accepted, Bool.false_eq_true, ↓reduceIte] at tested
          obtain ⟨ready, allocated, prepared, stable⟩ := tested
          have getReady : ready "limit_mul" = some (number limit) :=
            (stable "limit_mul" (by decide)).trans (by simpa [bound, setLocal] using getLimit)
          have scanned := ih (first+1) (by omega) ready (heap ++ allocated) getReady
          simp only [List.range'_succ, List.find?_cons, accepted]
          cases selected : (List.range' (first+1) count).find? (fun x => ProjectEulerSolutions.PermutedMultiples.accepts limit x) with
          | some value =>
              rw [selected] at scanned
              obtain ⟨extra, returned⟩ := scanned
              refine ⟨allocated ++ extra, ?_⟩
              intro following
              apply BlockReturns.range_step active
              have rest := prepared.returns (returned following)
              simpa only [List.append_assoc, Nat.cast_add, Nat.cast_one] using rest
          | none =>
              rw [selected] at scanned
              obtain ⟨finalEnv, extra, continued, finalStable⟩ := scanned
              refine ⟨finalEnv, allocated ++ extra, ?_, ?_⟩
              · apply BlockContinues.range_step active
                simpa only [List.append_assoc, Nat.cast_add, Nat.cast_one] using prepared.append continued
              · exact search_stable_trans (search_stable_bind env first) (search_stable_trans stable finalStable)

open ProjectEulerSolutions.PermutedMultiples

private def boundsEnv (env : HeapCalls.Env) (limit digits : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "lo" (number (lower digits))) "hi" (number (upper limit digits))

private theorem ten_power (exponent : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (expression : Expr)
    (evaluated : Evaluates program env expression heap (number exponent) heap) :
    Evaluates program env (.binOp (.constant (.int 10)) .pow expression) heap (number (10 ^ exponent : Nat)) heap := by
  exact Evaluates.binary (av := number 10) (bv := number exponent) (middle := heap) (afterRight := heap)
    Evaluates.constant evaluated (by simp [Mutable.binary, scalarBinary, show ¬ (exponent : Int) < 0 by omega])

/-- The original width arithmetic computes the complete positive candidate interval. -/
theorem prepare_width (limit digits : Nat) (positive : 1 ≤ digits) (supported : 2 ≤ limit)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getLimit : env "limit_mul" = some (number limit))
    (getDigits : env "d" = some (number digits)) :
    BlockContinues program env (widthBody.take 2) heap (boundsEnv env limit digits) heap := by
  let withLower := setLocal env "lo" (number (lower digits))
  have first : BlockContinues program env [widthBody[0]'(by decide)] heap withLower heap := by
    apply BlockContinues.assign
    apply ten_power (digits-1)
    apply Evaluates.binary (Evaluates.lookup getDigits) Evaluates.constant
    have cast : ((digits-1 : Nat) : Int) = (digits : Int)-1 := by omega
    simp [Mutable.binary, scalarBinary, cast]
  have second : BlockContinues program withLower [widthBody[1]'(by decide)] heap (boundsEnv env limit digits) heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (10^digits-1 : Nat)) (bv := number limit) (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number (10^digits : Nat)) (bv := number 1) (middle := heap) (afterRight := heap)
      · exact ten_power digits withLower heap (.name "d") (Evaluates.lookup (by simpa [withLower, setLocal] using getDigits))
      · exact Evaluates.constant
      · have nonzero : 0 < 10^digits := Nat.pow_pos (by decide)
        have cast : ((10^digits-1 : Nat) : Int) = (10 : Int)^digits-1 := by
          have powerCast : ((10^digits : Nat) : Int) = (10 : Int)^digits := by simp
          omega
        simp [Mutable.binary, scalarBinary, cast]
    · exact Evaluates.lookup (by simpa [withLower, setLocal] using getLimit)
    · simp [Mutable.binary, scalarBinary, upper, number, show 0 < limit by omega, show limit ≠ 0 by omega]
  exact first.append second

private def WidthSearch (limit digits : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) : Option Nat → Prop
  | some value => ∃ extra, ∀ following, BlockReturns program env (widthBody.take 3 ++ following) heap (number value) (heap ++ extra)
  | none => ∃ finalEnv extra, BlockContinues program env (widthBody.take 3) heap finalEnv (heap ++ extra) ∧
      finalEnv "limit_mul" = some (number limit) ∧ finalEnv "d" = some (number digits)

/-- A full width iteration prepares its bounds and returns the first accepted candidate. -/
theorem width_result (limit digits : Nat) (supported : 2 ≤ limit ∧ limit ≤ 6) (positive : 1 ≤ digits)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getLimit : env "limit_mul" = some (number limit))
    (getDigits : env "d" = some (number digits)) :
    WidthSearch limit digits env heap
      ((List.range' (lower digits) (upper limit digits+1-lower digits)).find? (fun x => accepts limit x)) := by
  have prepared := prepare_width limit digits positive supported.1 env heap getLimit getDigits
  have startBound := window_nonempty limit digits supported positive
  have getReady : (boundsEnv env limit digits) "limit_mul" = some (number limit) := by simpa [boundsEnv, setLocal] using getLimit
  have scanned := candidates_loop limit (lower digits) (upper limit digits+1-lower digits) (upper limit digits+1)
    supported.1 (by omega) (boundsEnv env limit digits) heap getReady
  have arguments : Evaluates.Arguments program (boundsEnv env limit digits)
      [.name "lo", .binOp (.name "hi") .add (.constant (.int 1))] heap
      [number (lower digits), number (upper limit digits+1 : Nat)] heap := by
    refine .cons (Evaluates.lookup (by simp [boundsEnv, setLocal])) (.cons ?_ .nil)
    exact Evaluates.binary (av := number (upper limit digits)) (bv := number 1) (middle := heap) (afterRight := heap)
      (Evaluates.lookup (by simp [boundsEnv, setLocal])) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  cases selected : (List.range' (lower digits) (upper limit digits+1-lower digits)).find? (fun x => accepts limit x) with
  | some value =>
      rw [selected] at scanned
      obtain ⟨extra, returned⟩ := scanned
      refine ⟨extra, ?_⟩
      intro following
      exact prepared.returns (BlockReturns.for_range arguments rfl (returned following))
  | none =>
      rw [selected] at scanned
      obtain ⟨finalEnv, extra, continued, stable⟩ := scanned
      refine ⟨finalEnv, extra, prepared.append (BlockContinues.for_range arguments rfl continued), ?_, ?_⟩
      · exact (stable "limit_mul" (by decide)).trans getReady
      · exact (stable "d" (by decide)).trans (by simpa [boundsEnv, setLocal] using getDigits)

/-- The unbounded source width loop reaches the least positive reference answer within six widths. -/
private theorem widths_loop (limit digits : Nat) (supported : 2 ≤ limit ∧ limit ≤ 6) (positive : 1 ≤ digits)
    (before : digits ≤ answerWidth limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit_mul" = some (number limit)) (getDigits : env "d" = some (number digits)) :
    ∃ extra, BlockReturns program env [.whileDo (.constant (.bool true)) widthBody []]
      heap (number (answer limit)) (heap ++ extra) := by
  have interval : 2 ≤ (limit : Int) ∧ (limit : Int) ≤ 6 := by omega
  have scanned := width_result limit digits supported positive env heap getLimit getDigits
  have selected := window limit digits interval before
  by_cases inside : answer limit ≤ upper limit digits
  · rw [if_pos inside] at selected
    rw [selected] at scanned
    obtain ⟨extra, returned⟩ := scanned
    refine ⟨extra, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    exact returned (widthBody.drop 3 ++ [.whileDo (.constant (.bool true)) widthBody []])
  · rw [if_neg inside] at selected
    rw [selected] at scanned
    obtain ⟨ready, allocated, prepared, finalLimit, finalDigits⟩ := scanned
    have earlier : digits < answerWidth limit := by
      have reached := inside_width limit interval
      by_contra late
      have same : digits = answerWidth limit := by omega
      exact inside (same ▸ reached)
    let next := setLocal ready "d" (number (digits+1 : Nat))
    obtain ⟨extra, returned⟩ := widths_loop limit (digits+1) supported (by omega) (by omega) next (heap ++ allocated)
      (by simpa [next, setLocal] using finalLimit) (by simp [next, setLocal])
    refine ⟨allocated ++ extra, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply prepared.returns
    apply BlockReturns.augAssign (value := number (digits+1 : Nat)) (middle := heap ++ allocated) (afterOperation := heap ++ allocated)
      finalDigits Evaluates.constant
    · simp [inplace, Mutable.binary, scalarBinary]
    · simpa only [List.append_assoc] using returned
termination_by answerWidth limit - digits
decreasing_by omega

/-- The executable optimized Python solver terminates with the least positive reference answer. -/
theorem result (limit : Nat) (supported : 2 ≤ limit ∧ limit ≤ 6) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [number limit] heap (number (answer limit)) (heap ++ extra) := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number limit] []
  obtain ⟨extra, scanned⟩ := widths_loop limit 1 supported (by decide)
    (width_bounds limit (by omega)).1 (setLocal entry "d" (number 1)) heap (by simp [setLocal]; rfl) (by simp [setLocal])
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  exact scanned

private abbrev inputEnv (limit : Int) : ScalarEnv :=
  fun name => if name = "limit_mul" then some (.int limit) else none

private theorem target_bounds (limit : Int)
    (lower : Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_0)
    (upper : Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_1) : 2 ≤ limit ∧ limit ≤ 6 := by
  constructor
  · simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare lower rfl
  · simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare upper rfl

theorem «Equiv(naive)» (limit : Int)
    (lower : Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_0)
    (upper : Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_1) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv (HeapCalls.scalarFunction P52Ast.module "naive" [.int limit]) := by
  have supported := target_bounds limit lower upper
  lift limit to Nat using (by omega : 0 ≤ limit)
  obtain ⟨extra, returned⟩ := result limit (by omega) []
  obtain ⟨referenceExtra, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p52.naive.result limit supported []
  have optimized := returned.scalar_finishes (values := [.int limit])
  have original := reference.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_0 →
      Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_1 →
      (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_0 →
      Calls.Condition program (inputEnv limit) Solvers.P52Ast.requires_solve_1 →
      (HeapCalls.scalarFunction P52Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit lower upper
    have supported := target_bounds limit lower upper
    lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨extra, returned⟩ := result limit (by omega) []
    exact (returned.scalar_returns (values := [.int limit])).supported
  · intro limit lower upper
    obtain ⟨extra, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p52.naive.result
      limit (target_bounds limit lower upper) []
    exact (returned.scalar_returns (values := [.int limit])).supported

end solve

end solvers.«52»
