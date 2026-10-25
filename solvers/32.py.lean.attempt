import «.formalsw».solvers.«32.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.SetSumCalls
import ProjectEulerStatements.Lib.PandigitalResult
import data.«project-euler-statements».data.py.project_euler_statements.«p32.py»

namespace solvers.«32»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.PandigitalProducts
private abbrev program := Solvers.P32Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev pair (result : Int × Nat) : Mutable.Object :=
  .tuple [.scalar (.int result.1), number result.2]
private abbrev digitCode : List Stmt :=
  executableBody ((findFunctionIn program.body "digit_mask").getD ([], [])).2

set_option Elab.async false

/-- The helper allocates only the final tuple; intermediate masks remain integer locals. -/
private theorem digit_loop (n mask length : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals)
    (getN : env "n" = some (number n))
    (getMask : env "mask" = some (number mask))
    (getLength : env "length" = some (number length)) :
    BlockReturns program env (digitCode.drop 2) heap (.ref heap.length)
      (heap ++ [pair (digitMaskAux n mask length)]) globals := by
  by_cases positive : 0 < n
  · rw [digitMaskAux_step n mask length positive]
    apply BlockReturns.while_loop (flag := true)
      (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, positive])) rfl
    apply BlockReturns.assign (value := number (n % 10)) (middle := heap)
    · apply Evaluates.binary (av := number n) (bv := number 10)
        (Evaluates.lookup getN) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    · let digitEnv := setLocal env "d" (number (n % 10))
      have getD : digitEnv "d" = some (number (n % 10)) := by simp [digitEnv, setLocal]
      by_cases zero : n % 10 = 0
      · rw [if_pos zero]
        apply BlockReturns.branch (flag := true)
          (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])) rfl
        apply BlockReturns.return_value
        exact Evaluates.tuple (.cons (Evaluates.unary Evaluates.constant rfl)
          (.cons Evaluates.constant .nil)) (by simp)
      · rw [if_neg zero]
        apply BlockReturns.branch (flag := false)
          (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
            (Evaluates.lookup getD) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
        let bit := 2 ^ (n % 10 - 1)
        let bitEnv := setLocal digitEnv "bit" (number bit)
        have getBit : bitEnv "bit" = some (number bit) := by simp [bitEnv, setLocal]
        have oldMask : bitEnv "mask" = some (number mask) := by
          simpa [bitEnv, digitEnv, setLocal] using getMask
        apply BlockReturns.assign (value := number bit) (middle := heap)
        · apply Evaluates.binary (av := number 1) (bv := number (n % 10 - 1))
            (middle := heap) (afterRight := heap)
          · exact Evaluates.constant
          · apply Evaluates.binary (av := number (n % 10)) (bv := number 1)
              (Evaluates.lookup getD) Evaluates.constant
            simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (show 1 ≤ n % 10 by omega)]
          · simp [Mutable.binary, scalarShiftLeft, number, integerShiftLeft, bit,
              show ¬ ((n % 10 - 1 : Nat) : Int) < 0 by omega]
        · have masked : Evaluates program bitEnv (.binOp (.name "mask") .bitAnd (.name "bit"))
              heap (number (mask &&& bit)) heap globals := by
            exact Evaluates.binary (Evaluates.lookup oldMask) (Evaluates.lookup getBit)
              (by simp [Mutable.binary, scalarBitAnd, number])
          by_cases overlap : mask &&& bit ≠ 0
          · rw [if_pos overlap]
            apply BlockReturns.branch (flag := true) masked
              (by simp [Mutable.truth, scalarTruth, overlap])
            apply BlockReturns.return_value
            exact Evaluates.tuple (.cons (Evaluates.unary Evaluates.constant rfl)
              (.cons Evaluates.constant .nil)) (by simp)
          · rw [if_neg overlap]
            have empty : mask &&& bit = 0 := by omega
            apply BlockReturns.branch (flag := false) masked
              (by simp [Mutable.truth, scalarTruth, empty])
            apply BlockReturns.augAssign (left := number mask) (right := number bit)
              (value := number (mask ||| bit)) (middle := heap) (afterOperation := heap) oldMask
              (Evaluates.lookup getBit)
            · simp [inplace, Mutable.binary, scalarBitOr, number]
            · apply BlockReturns.augAssign (left := number length) (right := number 1)
                (value := number (length + 1)) (middle := heap) (afterOperation := heap)
                (by simpa [setLocal, bitEnv, digitEnv] using getLength) Evaluates.constant
              · simp [inplace, Mutable.binary, scalarBinary, number]
              · apply BlockReturns.augAssign (left := number n) (right := number 10)
                  (value := number (n / 10)) (middle := heap) (afterOperation := heap)
                  (by simpa [setLocal, bitEnv, digitEnv] using getN) Evaluates.constant
                · simp [inplace, Mutable.binary, scalarBinary, number]
                · have rest := digit_loop (n / 10) (mask ||| bit) (length + 1) heap
                    (setLocal (setLocal (setLocal bitEnv "mask" (number (mask ||| bit)))
                      "length" (number (length + 1))) "n" (number (n / 10))) globals
                    (by simp [setLocal]) (by simp [setLocal]) (by simp [setLocal])
                  simpa [digitCode, program, Solvers.P32Ast.module, findFunctionIn, executableBody,
                    Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, bitEnv, digitEnv, bit] using rest
  · have zero : n = 0 := by omega
    subst n
    rw [digitMaskAux_zero]
    apply BlockReturns.while_loop (flag := false)
      (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])) rfl
    apply BlockReturns.return_value
    exact Evaluates.tuple (.cons (Evaluates.lookup getMask) (.cons (Evaluates.lookup getLength) .nil)) (by simp)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by decide)

/-- This helper theorem is used only by solve's immediately unpacked calls. -/
theorem digit_mask.result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "digit_mask" [number n] heap (.ref heap.length)
      (heap ++ [pair (digitMask n)]) globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply digit_loop n 0 0 heap _ globals
  · cases globals <;> rfl
  · rfl
  · rfl

/- A single source loop iteration, stopping at its pending continuation. This
relation preserves Python continue without rerunning the skipped source tail. -/
private def Iterates (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∀ next following anchors stack,
    Reaches program anchors
      ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals

private theorem Iterates.done (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    Iterates env [] heap env heap globals := by
  intro next following anchors stack
  exact .next rfl .refl

private theorem Iterates.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap globals)
    (rest : Iterates middleEnv tail middleHeap finalEnv finalHeap globals) :
    Iterates env (body ++ tail) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have tailRun : Reaches program anchors
      ⟨.block (tail ++ (.loopResume next :: following)) middleEnv, stack, middleHeap⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals := by
    simpa only [List.append_assoc, List.singleton_append] using rest next following anchors stack
  simpa only [List.append_assoc, List.singleton_append] using
    (first (tail ++ (.loopResume next :: following)) anchors stack).trans tailRun

private theorem Iterates.branch
    (evaluated : Evaluates program env condition heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Iterates env ((if flag then yes else no) ++ body) middle finalEnv finalHeap globals) :
    Iterates env (.ifThen condition yes no :: body) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  apply Reaches.next rfl
  apply (evaluated anchors _).trans
  apply Reaches.next (following :=
    ⟨.block (((if flag then yes else no) ++ body) ++ [.loopResume next] ++ following) env, stack, middle⟩)
    (by simp [HeapCalls.step, run, truth, List.append_assoc])
  exact rest next following anchors stack

private theorem Iterates.skip
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    Iterates env (.continueLoop :: body) heap env heap globals := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private theorem Iterates.then_loop
    (iteration : Iterates env body heap middleEnv middleHeap globals)
    (rest : BlockContinues program middleEnv [next] middleHeap finalEnv finalHeap globals) :
    BlockContinues program env (body ++ [.loopResume next]) heap finalEnv finalHeap globals := by
  intro following anchors stack
  exact (iteration next following anchors stack).trans (rest following anchors stack)

private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private def inserted (before : List Constant) (values : List Nat) : List Constant :=
  values.foldl (fun state n => ScalarSet.insert state (.int n)) before

private structure SearchLocals (env : HeapCalls.Env) (address : Nat) : Prop where
  products : env "products" = some (.ref address)
  full : env "FULL_MASK" = some (.scalar (.int 511))

private theorem SearchLocals.other (locals : SearchLocals env address)
    (name : String) (value : Mutable.Value) (notProducts : name ≠ "products") (notFull : name ≠ "FULL_MASK") :
    SearchLocals (setLocal env name value) address := by
  constructor
  · simpa [setLocal, Ne.symm notProducts] using locals.products
  · simpa [setLocal, Ne.symm notFull] using locals.full

private structure InnerLocals (env : HeapCalls.Env) (a address : Nat) : Prop where
  shared : SearchLocals env address
  aValue : env "a" = some (number a)
  aMask : env "ma" = some (.scalar (.int (digitMask a).1))
  aLength : env "la" = some (number (digitMask a).2)

private theorem InnerLocals.other (locals : InnerLocals env a address)
    (name : String) (value : Mutable.Value)
    (unrelated : name ∉ ["products", "FULL_MASK", "a", "ma", "la"]) :
    InnerLocals (setLocal env name value) a address := by
  rcases locals with ⟨⟨products, full⟩, aValue, aMask, aLength⟩
  simp only [List.mem_cons, not_or] at unrelated
  constructor
  · constructor <;> simp_all [setLocal, eq_comm]
  · simp_all [setLocal, eq_comm]
  · simp_all [setLocal, eq_comm]
  · simp_all [setLocal, eq_comm]

private theorem unpack_mask (n : Nat) (input left right : String) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getN : env input = some (number n)) :
    BlockContinues program env
      [.assign [.tuple [.name left, .name right]] (.call (.name "digit_mask") [.name input])]
      heap (assignNames env [left, right] [.scalar (.int (digitMask n).1), number (digitMask n).2])
      (heap ++ [pair (digitMask n)]) globals := by
  apply BlockContinues.unpack (names := [left, right])
    (items := [.scalar (.int (digitMask n).1), number (digitMask n).2]) rfl
    (Evaluates.call (.cons (Evaluates.lookup getN) .nil) (digit_mask.result n heap globals))
  · simp [Mutable.sequence, pair]
  · rfl

private def workHeap (base : Mutable.Heap) (values : List Constant) (garbage : Mutable.Heap) : Mutable.Heap :=
  base ++ (.set values :: garbage)

private theorem workHeap_found (base : Mutable.Heap) (values : List Constant) (garbage : Mutable.Heap) :
    (workHeap base values garbage)[base.length]? = some (.set values) := by simp [workHeap]

private theorem workHeap_append (base : Mutable.Heap) (values : List Constant) (garbage extra : Mutable.Heap) :
    workHeap base values garbage ++ extra = workHeap base values (garbage ++ extra) := by
  simp [workHeap, List.append_assoc]

private theorem workHeap_set (base : Mutable.Heap) (values replacement : List Constant) (garbage : Mutable.Heap) :
    (workHeap base values garbage).set base.length (.set replacement) = workHeap base replacement garbage := by
  simp [workHeap, List.set_append_right]

private abbrev solveCode : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev firstOuter : List Stmt :=
  match (solveCode[1]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev candidateBody : List Stmt :=
  match (firstOuter[6]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []

/-- The disjunction returns either the negative-mask test or the actual integer overlap. -/
private theorem overlap_guard (mask : Nat) (other : Int) (left : Expr) (right : String)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (leftValue : Evaluates program env left heap (number mask) heap globals)
    (getOther : env right = some (.scalar (.int other))) :
    ∃ value, Evaluates program env
      (.boolOp .or [.compare (.name right) [(.lt, .constant (.int 0))],
        .binOp left .bitAnd (.name right)]) heap value heap globals ∧
      Mutable.truth value heap = some (.ok (decide (other < 0 ∨ mask &&& other.toNat ≠ 0)), heap) := by
  by_cases negative : other < 0
  · refine ⟨.scalar (.bool true), ?_, ?_⟩
    · exact Evaluates.boolean_stop (flag := true)
        (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
          (Evaluates.lookup getOther) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative])) rfl rfl
    · simp [Mutable.truth, scalarTruth, negative]
  · have nonnegative : 0 ≤ other := by omega
    have represented : ((other.toNat : Nat) : Int) = other := Int.toNat_of_nonneg nonnegative
    refine ⟨number (mask &&& other.toNat), ?_, ?_⟩
    · apply Evaluates.boolean_next (flag := false)
        (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
          (Evaluates.lookup getOther) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative])) rfl rfl
      apply Evaluates.boolean_last (flag := (((mask &&& other.toNat : Nat) : Int) != 0))
      · have readNat : env right = some (number other.toNat) := by
          simpa only [number, represented] using getOther
        exact Evaluates.binary leftValue (Evaluates.lookup readNat) rfl
      · rfl
    · cases equal : mask &&& other.toNat with
      | zero => simp [Mutable.truth, scalarTruth, negative]
      | succ k => simp [Mutable.truth, scalarTruth, negative]; omega

private theorem outside_guard (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getP : env "p" = some (number n)) :
    Evaluates program env
      (.boolOp .or [.compare (.name "p") [(.lt, .constant (.int 1000))],
        .compare (.name "p") [(.gt, .constant (.int 9999))]])
      heap (.scalar (.bool (decide (n < 1000 ∨ 9999 < n)))) heap globals := by
  have first : Evaluates program env (.compare (.name "p") [(.lt, .constant (.int 1000))])
      heap (.scalar (.bool (decide (n < 1000)))) heap globals := by
    exact Evaluates.compare (Evaluates.lookup getP) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases small : n < 1000
  · have yes : decide (n < 1000 ∨ 9999 < n) = true := by simp [small]
    rw [yes]
    exact Evaluates.boolean_stop (flag := true) (by simpa [small] using first) rfl rfl
  · have reduced : decide (n < 1000 ∨ 9999 < n) = decide (9999 < n) := by simp [small]
    rw [reduced]
    apply Evaluates.boolean_next (flag := false) (by simpa [small] using first) rfl rfl
    apply Evaluates.boolean_last
    · exact Evaluates.compare (Evaluates.lookup getP) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    · rfl

private theorem final_guard (ma mb mp la lb lp : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getMa : env "ma" = some (number ma)) (getMb : env "mb" = some (number mb))
    (getMp : env "mp" = some (number mp)) (getLa : env "la" = some (number la))
    (getLb : env "lb" = some (number lb)) (getLp : env "lp" = some (number lp))
    (getFull : env "FULL_MASK" = some (number 511)) :
    Evaluates program env
      (.boolOp .and [
        .compare (.binOp (.binOp (.name "la") .add (.name "lb")) .add (.name "lp")) [(.eq, .constant (.int 9))],
        .compare (.binOp (.binOp (.name "ma") .bitOr (.name "mb")) .bitOr (.name "mp")) [(.eq, .name "FULL_MASK")]])
      heap (.scalar (.bool (decide (la + lb + lp = 9 ∧ ma ||| mb ||| mp = 511)))) heap globals := by
  have summed : Evaluates program env
      (.binOp (.binOp (.name "la") .add (.name "lb")) .add (.name "lp")) heap
      (number (la + lb + lp)) heap globals :=
    Evaluates.binary (av := number (la + lb)) (bv := number lp) (middle := heap) (afterRight := heap)
      (Evaluates.binary (value := number (la + lb)) (finalHeap := heap)
        (Evaluates.lookup getLa) (Evaluates.lookup getLb)
        (by simp [Mutable.binary, scalarBinary, number]))
      (Evaluates.lookup getLp) (by simp [Mutable.binary, scalarBinary, number])
  have first : Evaluates program env
      (.compare (.binOp (.binOp (.name "la") .add (.name "lb")) .add (.name "lp")) [(.eq, .constant (.int 9))])
      heap (.scalar (.bool (decide (la + lb + lp = 9)))) heap globals := by
    exact Evaluates.compare summed Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  by_cases length : la + lb + lp = 9
  · have reduced : decide (la + lb + lp = 9 ∧ ma ||| mb ||| mp = 511) = decide (ma ||| mb ||| mp = 511) := by
      simp [length]
    rw [reduced]
    apply Evaluates.boolean_next (flag := true) (by simpa [length] using first) rfl rfl
    apply Evaluates.boolean_last
    · apply Evaluates.compare
        (Evaluates.binary (av := number (ma ||| mb)) (bv := number mp)
          (value := number (ma ||| mb ||| mp)) (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.binary (value := number (ma ||| mb)) (finalHeap := heap)
            (Evaluates.lookup getMa) (Evaluates.lookup getMb) rfl)
          (Evaluates.lookup getMp) rfl)
        (Evaluates.lookup getFull)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      omega
    · rfl
  · have no : decide (la + lb + lp = 9 ∧ ma ||| mb ||| mp = 511) = false := by simp [length]
    rw [no]
    exact Evaluates.boolean_stop (flag := false) (by simpa [length] using first) rfl rfl

/-- One candidate preserves the live product set except for a proven insertion. -/
private theorem candidate (a b : Nat) (base : Mutable.Heap) (values : List Constant)
    (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (locals : InnerLocals env a base.length) (getB : env "b" = some (number b))
    (maNonnegative : 0 ≤ (digitMask a).1) :
    ∃ finalEnv finalGarbage, InnerLocals finalEnv a base.length ∧
      Iterates env candidateBody (workHeap base values garbage) finalEnv
        (workHeap base (if accepted a b then ScalarSet.insert values (.int (a * b)) else values)
          finalGarbage) globals := by
  let afterB := setLocal (setLocal env "mb" (.scalar (.int (digitMask b).1))) "lb" (number (digitMask b).2)
  let garbageB := garbage ++ [pair (digitMask b)]
  have localsB : InnerLocals afterB a base.length :=
    (locals.other "mb" (.scalar (.int (digitMask b).1)) (by decide)).other "lb" (number (digitMask b).2) (by decide)
  have unpackB := unpack_mask b "b" "mb" "lb" (workHeap base values garbage) env globals getB
  rw [workHeap_append] at unpackB
  have maCast : ((digitMask a).1.toNat : Int) = (digitMask a).1 := Int.toNat_of_nonneg maNonnegative
  have readMa : afterB "ma" = some (number (digitMask a).1.toNat) := by
    simpa only [number, maCast] using localsB.aMask
  obtain ⟨bValue, bGuard, bTruth⟩ := overlap_guard (digitMask a).1.toNat (digitMask b).1
    (.name "ma") "mb" afterB (workHeap base values garbageB) globals (Evaluates.lookup readMa)
    (by simp [afterB, setLocal])
  by_cases rejectB : (digitMask b).1 < 0 ∨ (digitMask a).1.toNat &&& (digitMask b).1.toNat ≠ 0
  · have rejected : accepted a b = false := by
      simp only [accepted, decide_eq_false_iff_not]
      omega
    refine ⟨afterB, garbageB, localsB, ?_⟩
    simp only [rejected, Bool.false_eq_true, ↓reduceIte]
    apply Iterates.prepend unpackB
    apply Iterates.branch bGuard (by simpa [rejectB] using bTruth)
    exact Iterates.skip (by intro next following; rfl)
  · have mbNonnegative : 0 ≤ (digitMask b).1 := by omega
    have disjointB : (digitMask a).1.toNat &&& (digitMask b).1.toNat = 0 := by omega
    let afterP := setLocal afterB "p" (number (a * b))
    have localsP : InnerLocals afterP a base.length := localsB.other "p" (number (a * b)) (by decide)
    have multiply : BlockContinues program afterB
        [.assign [.name "p"] (.binOp (.name "a") .mult (.name "b"))]
        (workHeap base values garbageB) afterP (workHeap base values garbageB) globals := by
      apply BlockContinues.assign
      exact Evaluates.binary (Evaluates.lookup localsB.aValue)
        (Evaluates.lookup (by simpa [afterB, setLocal] using getB))
        (by simp [Mutable.binary, scalarBinary, number])
    have pGuard := outside_guard (a * b) afterP (workHeap base values garbageB) globals (by simp [afterP, setLocal])
    by_cases rejectP : a * b < 1000 ∨ 9999 < a * b
    · have rejected : accepted a b = false := by
        simp only [accepted, decide_eq_false_iff_not]
        omega
      refine ⟨afterP, garbageB, localsP, ?_⟩
      simp only [rejected, Bool.false_eq_true, ↓reduceIte]
      apply Iterates.prepend unpackB
      apply Iterates.branch bGuard (by simpa [rejectB] using bTruth)
      apply Iterates.prepend multiply
      apply Iterates.branch (flag := true) (by simpa [rejectP] using pGuard) rfl
      exact Iterates.skip (by intro next following; rfl)
    · have productLower : 1000 ≤ a * b := by omega
      have productUpper : a * b ≤ 9999 := by omega
      let afterM := setLocal (setLocal afterP "mp" (.scalar (.int (digitMask (a * b)).1)))
        "lp" (number (digitMask (a * b)).2)
      let garbageM := garbageB ++ [pair (digitMask (a * b))]
      have localsM : InnerLocals afterM a base.length :=
        (localsP.other "mp" (.scalar (.int (digitMask (a * b)).1)) (by decide)).other
          "lp" (number (digitMask (a * b)).2) (by decide)
      have unpackM := unpack_mask (a * b) "p" "mp" "lp" (workHeap base values garbageB)
        afterP globals (by simp [afterP, setLocal])
      rw [workHeap_append] at unpackM
      have mbCast : ((digitMask b).1.toNat : Int) = (digitMask b).1 := Int.toNat_of_nonneg mbNonnegative
      have readMaM : afterM "ma" = some (number (digitMask a).1.toNat) := by
        simpa only [number, maCast] using localsM.aMask
      have readMbM : afterM "mb" = some (number (digitMask b).1.toNat) := by
        simp [afterM, afterP, afterB, setLocal, number, mbCast]
      have combined : Evaluates program afterM (.binOp (.name "ma") .bitOr (.name "mb"))
          (workHeap base values garbageM) (number ((digitMask a).1.toNat ||| (digitMask b).1.toNat))
          (workHeap base values garbageM) globals := by
        exact Evaluates.binary (Evaluates.lookup readMaM) (Evaluates.lookup readMbM) rfl
      obtain ⟨mValue, mGuard, mTruth⟩ := overlap_guard ((digitMask a).1.toNat ||| (digitMask b).1.toNat)
        (digitMask (a * b)).1 (.binOp (.name "ma") .bitOr (.name "mb")) "mp" afterM
        (workHeap base values garbageM) globals combined (by simp [afterM, setLocal])
      by_cases rejectM : (digitMask (a * b)).1 < 0 ∨
          ((digitMask a).1.toNat ||| (digitMask b).1.toNat) &&& (digitMask (a * b)).1.toNat ≠ 0
      · have rejected : accepted a b = false := by
          simp only [accepted, decide_eq_false_iff_not]
          omega
        refine ⟨afterM, garbageM, localsM, ?_⟩
        simp only [rejected, Bool.false_eq_true, ↓reduceIte]
        apply Iterates.prepend unpackB
        apply Iterates.branch bGuard (by simpa [rejectB] using bTruth)
        apply Iterates.prepend multiply
        apply Iterates.branch (flag := false) (by simpa [rejectP] using pGuard) rfl
        apply Iterates.prepend unpackM
        apply Iterates.branch mGuard (by simpa [rejectM] using mTruth)
        exact Iterates.skip (by intro next following; rfl)
      · have mpNonnegative : 0 ≤ (digitMask (a * b)).1 := by omega
        have disjointM : ((digitMask a).1.toNat ||| (digitMask b).1.toNat) &&& (digitMask (a * b)).1.toNat = 0 := by omega
        have mpCast : ((digitMask (a * b)).1.toNat : Int) = (digitMask (a * b)).1 := Int.toNat_of_nonneg mpNonnegative
        let lastTest := (digitMask a).2 + (digitMask b).2 + (digitMask (a * b)).2 = 9 ∧
          ((digitMask a).1.toNat ||| (digitMask b).1.toNat ||| (digitMask (a * b)).1.toNat) = 511
        have exactTest : accepted a b = decide lastTest := by
          simp [accepted, lastTest, maNonnegative, mbNonnegative, disjointB, productLower, productUpper,
            mpNonnegative, disjointM]
        have lastGuard := final_guard (digitMask a).1.toNat (digitMask b).1.toNat (digitMask (a * b)).1.toNat
          (digitMask a).2 (digitMask b).2 (digitMask (a * b)).2 afterM (workHeap base values garbageM) globals
          readMaM readMbM (by simp [afterM, setLocal, number, mpCast]) localsM.aLength
          (by simp [afterM, afterP, afterB, setLocal]) (by simp [afterM, setLocal]) localsM.shared.full
        refine ⟨afterM, garbageM, localsM, ?_⟩
        apply Iterates.prepend unpackB
        apply Iterates.branch bGuard (by simpa [rejectB] using bTruth)
        apply Iterates.prepend multiply
        apply Iterates.branch (flag := false) (by simpa [rejectP] using pGuard) rfl
        apply Iterates.prepend unpackM
        apply Iterates.branch mGuard (by simpa [rejectM] using mTruth)
        by_cases matched : lastTest
        · simp only [exactTest, matched, decide_true, ↓reduceIte]
          apply Iterates.branch (flag := true) (by simpa [lastTest, matched] using lastGuard) rfl
          apply Iterates.prepend (body := [.expr (.call (.attribute (.name "products") "add") [.name "p"])])
          · apply BlockContinues.discard
            simpa only [workHeap_set] using Evaluates.add (value := .int (a * b)) (expression := .name "p")
              (Evaluates.lookup localsM.shared.products)
              (workHeap_found base values garbageM)
              (Evaluates.lookup (by simp [afterM, afterP, setLocal])) (workHeap_found base values garbageM)
          · exact Iterates.done _ _ _
        · simp only [exactTest, matched, decide_false]
          apply Iterates.branch (flag := false) (by simpa [lastTest, matched] using lastGuard) rfl
          exact Iterates.done _ _ _

private def rowSlice (a first count : Nat) : List Nat :=
  ((List.range' first count).filter (accepted a)).map (a * ·)

private theorem inner_scan (a first count stop : Nat) (bounds : stop - first = count)
    (base : Mutable.Heap) (values : List Constant) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (locals : InnerLocals env a base.length) (maNonnegative : 0 ≤ (digitMask a).1) :
    ∃ finalEnv finalGarbage, InnerLocals finalEnv a base.length ∧
      BlockContinues program env [.rangeStepLoop "b" first stop 1 candidateBody []]
        (workHeap base values garbage) finalEnv
        (workHeap base (inserted values (rowSlice a first count)) finalGarbage) globals := by
  induction count generalizing first values garbage env with
  | zero =>
      refine ⟨env, garbage, locals, ?_⟩
      simp only [rowSlice, List.range'_zero, List.filter_nil, List.map_nil, inserted]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let boundEnv := setLocal env "b" (number first)
      have boundLocals : InnerLocals boundEnv a base.length := locals.other "b" (number first) (by decide)
      obtain ⟨middleEnv, middleGarbage, middleLocals, pass⟩ := candidate a first base values garbage boundEnv
        globals boundLocals (by simp [boundEnv, setLocal]) maNonnegative
      let updated := if accepted a first then ScalarSet.insert values (.int (a * first)) else values
      obtain ⟨finalEnv, finalGarbage, finalLocals, rest⟩ := ih (first + 1) (by omega) updated middleGarbage
        middleEnv middleLocals
      have collected : inserted updated (rowSlice a (first + 1) count) = inserted values (rowSlice a first (count + 1)) := by
        cases selected : accepted a first <;>
          simp [rowSlice, List.range'_succ, selected, inserted, updated]
      rw [collected] at rest
      refine ⟨finalEnv, finalGarbage, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have controlled : Calls.loopControl candidateBody = true := rfl
      simpa only [Calls.loopContinuation, controlled, ↓reduceIte, Int.natCast_add, Int.natCast_one] using
        pass.then_loop rest

private def boundsBody (lower upper : Nat) : List Stmt :=
  [.assign [.name "b_min"] (.binOp
      (.binOp (.binOp (.constant (.int 1000)) .add (.name "a")) .sub (.constant (.int 1)))
      .floorDiv (.name "a")),
   .assign [.name "b_max"] (.binOp (.constant (.int 9999)) .floorDiv (.name "a")),
   .ifThen (.compare (.name "b_min") [(.lt, .constant (.int lower))])
      [.assign [.name "b_min"] (.constant (.int lower))] [],
   .ifThen (.compare (.name "b_max") [(.gt, .constant (.int upper))])
      [.assign [.name "b_max"] (.constant (.int upper))] []]

private theorem prepare_bounds (lower upper a : Nat) (positive : 0 < a)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) (address : Nat)
    (locals : InnerLocals env a address) :
    ∃ finalEnv, InnerLocals finalEnv a address ∧
      finalEnv "b_min" = some (number (lowerBound lower a)) ∧
      finalEnv "b_max" = some (number (upperBound upper a)) ∧
      BlockContinues program env (boundsBody lower upper) heap finalEnv heap globals := by
  let low := (1000 + a - 1) / a
  let high := 9999 / a
  let afterLow := setLocal env "b_min" (number low)
  let afterHigh := setLocal afterLow "b_max" (number high)
  let clippedLow := if low < lower then setLocal afterHigh "b_min" (number lower) else afterHigh
  let clippedHigh := if upper < high then setLocal clippedLow "b_max" (number upper) else clippedLow
  have afterLowLocals : InnerLocals afterLow a address := locals.other "b_min" (number low) (by decide)
  have afterHighLocals : InnerLocals afterHigh a address := afterLowLocals.other "b_max" (number high) (by decide)
  have clippedLowLocals : InnerLocals clippedLow a address := by
    dsimp [clippedLow]
    split
    · exact afterHighLocals.other "b_min" (number lower) (by decide)
    · exact afterHighLocals
  have clippedHighLocals : InnerLocals clippedHigh a address := by
    dsimp [clippedHigh]
    split
    · exact clippedLowLocals.other "b_max" (number upper) (by decide)
    · exact clippedLowLocals
  have lowLookup : clippedLow "b_min" = some (number (lowerBound lower a)) := by
    by_cases clipped : low < lower
    · have maximum : lowerBound lower a = lower := by
        exact Nat.max_eq_left (by dsimp [low] at clipped; omega)
      simp [clippedLow, clipped, setLocal, maximum]
    · have maximum : lowerBound lower a = low := by
        exact Nat.max_eq_right (by dsimp [low] at clipped; omega)
      simp [clippedLow, afterHigh, afterLow, clipped, setLocal, maximum]
  have highLookup : clippedLow "b_max" = some (number high) := by
    dsimp [clippedLow]
    split <;> simp [afterHigh, setLocal]
  have finalLow : clippedHigh "b_min" = some (number (lowerBound lower a)) := by
    dsimp [clippedHigh]
    split <;> simpa [setLocal] using lowLookup
  have finalHigh : clippedHigh "b_max" = some (number (upperBound upper a)) := by
    by_cases clipped : upper < high
    · have minimum : upperBound upper a = upper := Nat.min_eq_left (by omega)
      simp [clippedHigh, clipped, setLocal, minimum]
    · have minimum : upperBound upper a = high := Nat.min_eq_right (by omega)
      simpa [clippedHigh, clipped, minimum] using highLookup
  have assignLow : BlockContinues program env [(boundsBody lower upper)[0]'(by simp [boundsBody])] heap afterLow heap globals := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (1000 + a - 1)) (bv := number a)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number (1000 + a)) (bv := number 1)
        (middle := heap) (afterRight := heap)
      · exact Evaluates.binary (value := number (1000 + a)) (finalHeap := heap)
          Evaluates.constant (Evaluates.lookup locals.aValue) (by simp [Mutable.binary, scalarBinary, number])
      · exact Evaluates.constant
      · simp [Mutable.binary, scalarBinary, number]
        omega
    · exact Evaluates.lookup locals.aValue
    · simp [Mutable.binary, scalarBinary, number, low, show a ≠ 0 by omega, positive]
      rw [Int.add_ediv_of_dvd_right (Int.dvd_refl _)]
      rw [Int.ediv_self (show (a : Int) ≠ 0 by omega)]
  have assignHigh : BlockContinues program afterLow [(boundsBody lower upper)[1]'(by simp [boundsBody])] heap afterHigh heap globals := by
    apply BlockContinues.assign
    exact Evaluates.binary Evaluates.constant (Evaluates.lookup afterLowLocals.aValue)
      (by simp [Mutable.binary, scalarBinary, number, high, show a ≠ 0 by omega, positive])
  have clipLow : BlockContinues program afterHigh [(boundsBody lower upper)[2]'(by simp [boundsBody])] heap clippedLow heap globals := by
    by_cases clipped : low < lower
    · simp only [clippedLow, clipped, ↓reduceIte]
      apply BlockContinues.branch (flag := true)
        (Evaluates.compare (av := number low) (bv := number lower)
          (value := .scalar (.bool true)) (finalHeap := heap)
          (Evaluates.lookup (by simp [afterHigh, afterLow, setLocal])) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, clipped])) rfl
      exact BlockContinues.assign Evaluates.constant
    · simp only [clippedLow, clipped, ↓reduceIte]
      apply BlockContinues.branch (flag := false)
        (Evaluates.compare (av := number low) (bv := number lower)
          (value := .scalar (.bool false)) (finalHeap := heap)
          (Evaluates.lookup (by simp [afterHigh, afterLow, setLocal])) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, clipped])) rfl
      exact BlockContinues.nil
  have clipHigh : BlockContinues program clippedLow [(boundsBody lower upper)[3]'(by simp [boundsBody])] heap clippedHigh heap globals := by
    by_cases clipped : upper < high
    · simp only [clippedHigh, clipped, ↓reduceIte]
      apply BlockContinues.branch (flag := true)
        (Evaluates.compare (av := number high) (bv := number upper)
          (value := .scalar (.bool true)) (finalHeap := heap)
          (Evaluates.lookup highLookup) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, clipped])) rfl
      exact BlockContinues.assign Evaluates.constant
    · simp only [clippedHigh, clipped, ↓reduceIte]
      apply BlockContinues.branch (flag := false)
        (Evaluates.compare (av := number high) (bv := number upper)
          (value := .scalar (.bool false)) (finalHeap := heap)
          (Evaluates.lookup highLookup) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, clipped])) rfl
      exact BlockContinues.nil
  exact ⟨clippedHigh, clippedHighLocals, finalLow, finalHigh,
    assignLow.append (assignHigh.append (clipLow.append clipHigh))⟩

private def outerBody (lower upper : Nat) : List Stmt :=
  [.assign [.tuple [.name "ma", .name "la"]] (.call (.name "digit_mask") [.name "a"]),
   .ifThen (.compare (.name "ma") [(.lt, .constant (.int 0))]) [.continueLoop] []] ++
  boundsBody lower upper ++
  [.forIn (.name "b") (.call (.name "range")
    [.name "b_min", .binOp (.name "b_max") .add (.constant (.int 1))]) candidateBody []]

private theorem row_pass (lower upper a : Nat) (positive : 0 < a)
    (base : Mutable.Heap) (values : List Constant) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (locals : SearchLocals env base.length) (getA : env "a" = some (number a)) :
    ∃ finalEnv finalGarbage, SearchLocals finalEnv base.length ∧
      Iterates env (outerBody lower upper) (workHeap base values garbage) finalEnv
        (workHeap base (inserted values (row lower upper a)) finalGarbage) globals := by
  let afterA := setLocal (setLocal env "ma" (.scalar (.int (digitMask a).1))) "la" (number (digitMask a).2)
  let garbageA := garbage ++ [pair (digitMask a)]
  have localsA : InnerLocals afterA a base.length := by
    constructor
    · exact (locals.other "ma" (.scalar (.int (digitMask a).1)) (by decide) (by decide)).other
        "la" (number (digitMask a).2) (by decide) (by decide)
    · simpa [afterA, setLocal] using getA
    · simp [afterA, setLocal]
    · simp [afterA, setLocal]
  have unpackA := unpack_mask a "a" "ma" "la" (workHeap base values garbage) env globals getA
  rw [workHeap_append] at unpackA
  by_cases negative : (digitMask a).1 < 0
  · refine ⟨afterA, garbageA, localsA.shared, ?_⟩
    simp only [row, negative, ↓reduceIte, inserted]
    apply Iterates.prepend unpackA
    apply Iterates.branch (flag := true)
      (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := workHeap base values garbageA)
        (Evaluates.lookup localsA.aMask) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative, garbageA])) rfl
    exact Iterates.skip (by intro next following; rfl)
  · obtain ⟨prepared, preparedLocals, minimum, maximum, setup⟩ := prepare_bounds lower upper a positive
      (workHeap base values garbageA) afterA globals base.length localsA
    obtain ⟨finalEnv, finalGarbage, finalLocals, loop⟩ := inner_scan a (lowerBound lower a)
      (upperBound upper a + 1 - lowerBound lower a) (upperBound upper a + 1) rfl
      base values garbageA prepared globals preparedLocals (by omega)
    have innerRun : BlockContinues program prepared
        [.forIn (.name "b") (.call (.name "range")
          [.name "b_min", .binOp (.name "b_max") .add (.constant (.int 1))]) candidateBody []]
        (workHeap base values garbageA) finalEnv
        (workHeap base (inserted values (row lower upper a)) finalGarbage) globals := by
      apply BlockContinues.for_range (values := [number (lowerBound lower a), number (upperBound upper a + 1)])
        (middle := workHeap base values garbageA)
      · exact .cons (Evaluates.lookup minimum) (.cons
          (Evaluates.binary (Evaluates.lookup maximum) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number])) .nil)
      · rfl
      · simpa only [row, negative, ↓reduceIte, rowSlice] using loop
    refine ⟨finalEnv, finalGarbage, finalLocals.shared, ?_⟩
    apply Iterates.prepend unpackA
    apply Iterates.branch (flag := false)
      (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := workHeap base values garbageA)
        (Evaluates.lookup localsA.aMask) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, negative, garbageA])) rfl
    apply Iterates.prepend setup
    apply Iterates.prepend innerRun
    exact Iterates.done _ _ _

private theorem outer_scan (lower upper first count stop : Nat) (positive : 0 < first)
    (bounds : stop - first = count)
    (base : Mutable.Heap) (values : List Constant) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (locals : SearchLocals env base.length) :
    ∃ finalEnv finalGarbage, SearchLocals finalEnv base.length ∧
      BlockContinues program env [.rangeStepLoop "a" first stop 1 (outerBody lower upper) []]
        (workHeap base values garbage) finalEnv
        (workHeap base (inserted values ((List.range' first count).flatMap (row lower upper))) finalGarbage) globals := by
  induction count generalizing first values garbage env with
  | zero =>
      refine ⟨env, garbage, locals, ?_⟩
      simp only [List.range'_zero, List.flatMap_nil, inserted]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let boundEnv := setLocal env "a" (number first)
      have boundLocals : SearchLocals boundEnv base.length := locals.other "a" (number first) (by decide) (by decide)
      obtain ⟨middleEnv, middleGarbage, middleLocals, pass⟩ := row_pass lower upper first positive base values garbage
        boundEnv globals boundLocals (by simp [boundEnv, setLocal])
      obtain ⟨finalEnv, finalGarbage, finalLocals, rest⟩ := ih (first + 1) (by omega) (by omega)
        (inserted values (row lower upper first)) middleGarbage middleEnv middleLocals
      have collected : inserted (inserted values (row lower upper first))
          ((List.range' (first + 1) count).flatMap (row lower upper)) =
          inserted values ((List.range' first (count + 1)).flatMap (row lower upper)) := by
        simp [inserted, List.range'_succ, List.foldl_append]
      rw [collected] at rest
      refine ⟨finalEnv, finalGarbage, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have controlled : Calls.loopControl (outerBody lower upper) = true := rfl
      simpa only [Calls.loopContinuation, controlled, ↓reduceIte, Int.natCast_add, Int.natCast_one] using
        pass.then_loop rest

private def winners : List Int := [6952, 7852, 5796, 5346, 4396, 7254, 7632]

private theorem collected_products : inserted [] products = winners.map Constant.int := by
  rw [ProjectEulerStatements.PandigitalCertificate.optimized_products]
  decide +kernel

/-- Both source search cases construct the exact product set before summing in any order. -/
theorem solve.result (heap : Mutable.Heap) (globals : Mutable.Globals)
    (full : globals.lookup "FULL_MASK" = some (.scalar (.int 511))) :
    ∃ extra, SetCalls.ReturnsValue program "solve" [] heap (number 45228) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let initial := functionEnv args body [] globals
  let ready := setLocal initial "products" (.ref heap.length)
  have initialFull : initial "FULL_MASK" = some (number 511) := by
    have unbound : "FULL_MASK" ∉ blockLocals body := by
      simp [body, program, Solvers.P32Ast.module, findFunctionIn, blockLocals, assignedLocals, targetLocals]
    have noArgs : args = [] := rfl
    cases globals with
    | nil => cases full
    | cons entry rest =>
        simpa only [initial, functionEnv, noArgs, List.map_nil, List.not_mem_nil,
          ↓reduceIte, unbound] using full
  have readyLocals : SearchLocals ready heap.length :=
    ⟨rfl, by simpa [ready, setLocal] using initialFull⟩
  obtain ⟨afterFirst, garbageFirst, localsFirst, first⟩ := outer_scan 1000 9999 1 9 10 (by decide) rfl
    heap [] [] ready globals readyLocals
  obtain ⟨afterSecond, garbageSecond, localsSecond, second⟩ := outer_scan 100 999 10 90 100 (by decide) rfl
    heap (inserted [] ((List.range' 1 9).flatMap (row 1000 9999))) garbageFirst afterFirst globals localsFirst
  have joined : inserted (inserted [] ((List.range' 1 9).flatMap (row 1000 9999)))
      ((List.range' 10 90).flatMap (row 100 999)) = winners.map Constant.int := by
    rw [← collected_products]
    simp [products, inserted, List.foldl_append]
  rw [joined] at second
  have runFirst : BlockContinues program ready
      [.forIn (.name "a") (.call (.name "range") [.constant (.int 1), .constant (.int 10)])
        (outerBody 1000 9999) []]
      (workHeap heap [] []) afterFirst
      (workHeap heap (inserted [] ((List.range' 1 9).flatMap (row 1000 9999))) garbageFirst) globals :=
    BlockContinues.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl first
  have runSecond : BlockContinues program afterFirst
      [.forIn (.name "a") (.call (.name "range") [.constant (.int 10), .constant (.int 100)])
        (outerBody 100 999) []]
      (workHeap heap (inserted [] ((List.range' 1 9).flatMap (row 1000 9999))) garbageFirst) afterSecond
      (workHeap heap (winners.map Constant.int) garbageSecond) globals :=
    BlockContinues.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl second
  refine ⟨.set (winners.map Constant.int) :: garbageSecond, ?_⟩
  apply SetCalls.BlockReturns.function rfl rfl
  apply SetCalls.BlockReturns.after_heap (BlockContinues.annAssign
    (value := .ref heap.length) (finalHeap := workHeap heap [] []) (Evaluates.builtin .nil rfl))
  apply SetCalls.BlockReturns.after_heap runFirst
  apply SetCalls.BlockReturns.after_heap runSecond
  apply SetCalls.BlockReturns.return_value
  have summed := SetCalls.Evaluates.sum_integers (items := winners) (initial := 0)
    (module := program) (env := afterSecond) (globals := globals)
    (iterable := .name "products") (start := .constant (.int 0))
    (.cons (Evaluates.lookup localsSecond.products) (.cons Evaluates.constant .nil))
    (workHeap_found heap (winners.map Constant.int) garbageSecond)
  exact summed

private abbrev contractGlobals (mask : Int) : Mutable.Globals := [("FULL_MASK", .scalar (.int mask))]
private abbrev contractEnv (mask : Int) : HeapCalls.Env := fun name => (contractGlobals mask).lookup name

private theorem condition_result (mask : Int) :
    Evaluates program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (.scalar (.bool (decide (mask = 511)))) [] (contractGlobals mask) := by
  exact Evaluates.compare (av := .scalar (.int mask)) (bv := number 511)
    (Evaluates.lookup rfl) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private theorem accepted_mask (mask : Int)
    (requires : HeapCalls.Condition program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (contractGlobals mask)) : mask = 511 := by
  simpa using (condition_result mask).condition_iff.mp requires

theorem requires_solve_0_supported (mask : Int) :
    (HeapCalls.evaluate program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (contractGlobals mask)).Supported :=
  Execution.supported_of_finishes (condition_result mask).finishes

theorem solve.«Equiv(naive_default)» (mask : Int)
    (requires : HeapCalls.Condition program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (contractGlobals mask)) :
    (SetCalls.call program "solve" [] [] (contractGlobals mask)).Equiv
      (SetCalls.call P32Ast.module "naive_default" [] [] (contractGlobals mask)) := by
  have same := accepted_mask mask requires
  subst mask
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := solve.result [] (contractGlobals 511) rfl
  obtain ⟨rightHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p32.naive_default.result []
      (globals := contractGlobals 511)
  rw [reference_value] at reference
  apply Mutable.Call.equiv_of_finishes (optimized.finishes leftChoices) (reference.finishes rightChoices)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap (number 45228)
    (contractGlobals 511) (by simp [Mutable.Heap.Valid]) trivial (by simp [contractGlobals, Mutable.Value.edge, Graph.Edge.Within])

theorem solve.«Supported» :
    (∀ (mask : Int), HeapCalls.Condition program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (contractGlobals mask) → (SetCalls.call program "solve" [] [] (contractGlobals mask)).Supported) ∧
    (∀ (mask : Int), HeapCalls.Condition program (contractEnv mask) Solvers.P32Ast.requires_solve_0 []
      (contractGlobals mask) → (SetCalls.call P32Ast.module "naive_default" [] [] (contractGlobals mask)).Supported) := by
  constructor
  · intro mask requires choices
    have same := accepted_mask mask requires
    subst mask
    obtain ⟨_, optimized⟩ := solve.result [] (contractGlobals 511) rfl
    exact Execution.supported_of_finishes (optimized.finishes choices)
  · intro mask _ choices
    obtain ⟨_, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p32.naive_default.result []
        (globals := contractGlobals mask)
    exact Execution.supported_of_finishes (reference.finishes choices)

end solvers.«32»
