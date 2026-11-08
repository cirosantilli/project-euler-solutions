import solvers.«99.py.lean.d».Rounding

namespace solvers.«99».power_bounds
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals
set_option Elab.async false

abbrev values (bounds : Bounds) : List Mutable.Value :=
  [number bounds.lower, number bounds.upper, number bounds.shift]
abbrev accumulatorNames : List String := ["lower", "upper", "shift"]
abbrev factorNames : List String := ["factor_lower", "factor_upper", "factor_shift"]
abbrev roundCall (lower upper shift : Expr) : Expr :=
  .call (.name "round_power_bounds") [lower, upper, shift, .name "precision"]
abbrev accumulatorCode : List Stmt :=
  [.assign [.tuple (accumulatorNames.map Expr.name)]
    (roundCall (.binOp (.name "lower") .mult (.name "factor_lower"))
      (.binOp (.name "upper") .mult (.name "factor_upper"))
      (.binOp (.name "shift") .add (.name "factor_shift")))]
abbrev factorCode : List Stmt :=
  [.assign [.tuple (factorNames.map Expr.name)]
    (roundCall (.binOp (.name "factor_lower") .mult (.name "factor_lower"))
      (.binOp (.name "factor_upper") .mult (.name "factor_upper"))
      (.binOp (.constant (.int 2)) .mult (.name "factor_shift")))]
abbrev accumulatorBranch : Stmt :=
  .ifThen (.binOp (.name "exponent") .mod (.constant (.int 2))) accumulatorCode []
abbrev body : List Stmt :=
  [accumulatorBranch,
   .augAssign (.name "exponent") .floorDiv (.constant (.int 2)),
   .ifThen (.name "exponent") factorCode []]
abbrev loop : Stmt := .whileDo (.name "exponent") body []
abbrev returned : Stmt := .return (.tuple (accumulatorNames.map Expr.name))

theorem source_body : executableBody ((findFunctionIn program.body "power_bounds").getD ([], [])).2 =
    [.assign [.name "lower"] (.constant (.int 1)),
     .assign [.name "upper"] (.constant (.int 1)),
     .assign [.name "shift"] (.constant (.int 0)),
     .assign [.tuple (factorNames.map Expr.name)]
       (roundCall (.name "base") (.name "base") (.constant (.int 0))), loop, returned] := rfl

structure Locals (env : HeapCalls.Env) (precision remaining : Nat) (accumulator factor : Bounds) : Prop where
  getPrecision : env "precision" = some (number precision)
  getExponent : env "exponent" = some (number remaining)
  getLower : env "lower" = some (number accumulator.lower)
  getUpper : env "upper" = some (number accumulator.upper)
  getShift : env "shift" = some (number accumulator.shift)
  getFactorLower : env "factor_lower" = some (number factor.lower)
  getFactorUpper : env "factor_upper" = some (number factor.upper)
  getFactorShift : env "factor_shift" = some (number factor.shift)

theorem Locals.accumulator {env : HeapCalls.Env} {precision remaining : Nat} {a f : Bounds}
    (locals : Locals env precision remaining a f) (next : Bounds) :
    Locals (assignNames env accumulatorNames (values next)) precision remaining next f := by
  rcases locals with ⟨hp, he, hl, hu, hs, hfl, hfu, hfs⟩
  constructor <;> simp_all [assignNames, accumulatorNames, values, setLocal]

theorem Locals.factor {env : HeapCalls.Env} {precision remaining : Nat} {a f : Bounds}
    (locals : Locals env precision remaining a f) (next : Bounds) :
    Locals (assignNames env factorNames (values next)) precision remaining a next := by
  rcases locals with ⟨hp, he, hl, hu, hs, hfl, hfu, hfs⟩
  constructor <;> simp_all [assignNames, factorNames, values, setLocal]

theorem Locals.exponent {env : HeapCalls.Env} {precision remaining : Nat} {a f : Bounds}
    (locals : Locals env precision remaining a f) (next : Nat) :
    Locals (setLocal env "exponent" (number next)) precision next a f := by
  rcases locals with ⟨hp, he, hl, hu, hs, hfl, hfu, hfs⟩
  constructor <;> simp_all [setLocal]

theorem accumulator_step (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (precision remaining : Nat) (a f : Bounds) (locals : Locals env precision remaining a f) :
    BlockContinues program env accumulatorCode heap
      (assignNames env accumulatorNames (values (trim (multiply a f) precision)))
      (heap ++ [boundsObject (trim (multiply a f) precision)]) globals := by
  apply BlockContinues.unpack (items := values (trim (multiply a f) precision)) rfl
  · apply Evaluates.call _ (round_power_bounds.result (multiply a f) precision heap globals)
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getLower) (Evaluates.lookup locals.getFactorLower)
      simp [Mutable.binary, scalarBinary, number, multiply]
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getUpper) (Evaluates.lookup locals.getFactorUpper)
      simp [Mutable.binary, scalarBinary, number, multiply]
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getShift) (Evaluates.lookup locals.getFactorShift)
      simp [Mutable.binary, scalarBinary, number, multiply]
    exact .cons (Evaluates.lookup locals.getPrecision) .nil
  · simp [Mutable.sequence, boundsObject, values]
  · rfl

theorem factor_step (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (precision remaining : Nat) (a f : Bounds) (locals : Locals env precision remaining a f) :
    BlockContinues program env factorCode heap
      (assignNames env factorNames (values (trim (multiply f f) precision)))
      (heap ++ [boundsObject (trim (multiply f f) precision)]) globals := by
  apply BlockContinues.unpack (items := values (trim (multiply f f) precision)) rfl
  · apply Evaluates.call _ (round_power_bounds.result (multiply f f) precision heap globals)
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getFactorLower) (Evaluates.lookup locals.getFactorLower)
      simp [Mutable.binary, scalarBinary, number, multiply]
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getFactorUpper) (Evaluates.lookup locals.getFactorUpper)
      simp [Mutable.binary, scalarBinary, number, multiply]
    apply Evaluates.Arguments.cons (middle := heap)
    · apply Evaluates.binary (middle := heap) (afterRight := heap)
        Evaluates.constant (Evaluates.lookup locals.getFactorShift)
      simp [Mutable.binary, scalarBinary, number, multiply, two_mul]
    exact .cons (Evaluates.lookup locals.getPrecision) .nil
  · simp [Mutable.sequence, boundsObject, values]
  · rfl

def nextAccumulator (precision remaining : Nat) (a f : Bounds) : Bounds :=
  if remaining % 2 = 1 then trim (multiply a f) precision else a

def nextFactor (precision remaining : Nat) (f : Bounds) : Bounds :=
  if remaining / 2 = 0 then f else trim (multiply f f) precision

theorem step (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (precision remaining : Nat) (a f : Bounds) (locals : Locals env precision remaining a f) :
    ∃ nextEnv extra, BlockContinues program env body heap nextEnv (heap ++ extra) globals ∧
      Locals nextEnv precision (remaining / 2) (nextAccumulator precision remaining a f)
        (nextFactor precision remaining f) := by
  have first : ∃ nextEnv extra,
      BlockContinues program env [accumulatorBranch] heap nextEnv (heap ++ extra) globals ∧
      Locals nextEnv precision remaining (nextAccumulator precision remaining a f) f := by
    by_cases odd : remaining % 2 = 1
    · refine ⟨assignNames env accumulatorNames (values (trim (multiply a f) precision)),
        [boundsObject (trim (multiply a f) precision)], ?_, ?_⟩
      · apply BlockContinues.branch (flag := true) (middle := heap) (value := number (remaining % 2 : Nat))
        · apply Evaluates.binary (Evaluates.lookup locals.getExponent) Evaluates.constant
          simp [Mutable.binary, scalarBinary, number]
        · simp [Mutable.truth, scalarTruth, odd]
        · exact accumulator_step env heap globals precision remaining a f locals
      · simpa only [nextAccumulator, if_pos odd] using locals.accumulator (trim (multiply a f) precision)
    · have even : remaining % 2 = 0 := by omega
      refine ⟨env, [], ?_, ?_⟩
      · apply BlockContinues.branch (flag := false) (middle := heap) (value := number (remaining % 2 : Nat))
        · apply Evaluates.binary (Evaluates.lookup locals.getExponent) Evaluates.constant
          simp [Mutable.binary, scalarBinary, number]
        · simp [Mutable.truth, scalarTruth, even]
        · simpa using (BlockContinues.nil (module := program) (env := env) (heap := heap) (globals := globals))
      · simpa [nextAccumulator, odd] using locals
  obtain ⟨middleEnv, firstExtra, firstRan, middleLocals⟩ := first
  have divided := BlockContinues.augAssign (module := program) (globals := globals)
    (heap := heap ++ firstExtra) (op := BinOp.floorDiv) middleLocals.getExponent
    (Evaluates.constant (value := .int 2))
    (show inplace .floorDiv (number remaining) (number 2) (heap ++ firstExtra) =
      some (.ok (number (remaining / 2 : Nat)), heap ++ firstExtra) by
      simp [inplace, Mutable.binary, scalarBinary, number])
  have dividedLocals := middleLocals.exponent (remaining / 2)
  by_cases finished : remaining / 2 = 0
  · refine ⟨setLocal middleEnv "exponent" (number (remaining / 2 : Nat)), firstExtra, ?_, ?_⟩
    · apply firstRan.append
      apply divided.append
      apply BlockContinues.branch (flag := false) (Evaluates.lookup dividedLocals.getExponent)
        (by simp [Mutable.truth, scalarTruth, finished])
      exact BlockContinues.nil
    · simpa [nextFactor, finished] using dividedLocals
  · refine ⟨assignNames (setLocal middleEnv "exponent" (number (remaining / 2 : Nat)))
        factorNames (values (trim (multiply f f) precision)),
      firstExtra ++ [boundsObject (trim (multiply f f) precision)], ?_, ?_⟩
    · apply firstRan.append
      apply divided.append
      apply BlockContinues.branch (flag := true) (Evaluates.lookup dividedLocals.getExponent)
        (by
          simp [Mutable.truth, scalarTruth]
          change ¬((remaining / 2 : Nat) : Int) = 0
          exact_mod_cast finished)
      simpa only [List.append_assoc] using factor_step _ (heap ++ firstExtra) globals precision
        (remaining / 2) (nextAccumulator precision remaining a f) f dividedLocals
    · simpa [nextFactor, finished] using dividedLocals.factor (trim (multiply f f) precision)

theorem loop_result (precision remaining : Nat) (a f : Bounds)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env precision remaining a f) :
    ∃ extra, BlockReturns program env [loop, returned] heap (.ref (heap ++ extra).length)
      ((heap ++ extra) ++ [boundsObject (powerLoop precision a f remaining)]) globals := by
  by_cases finished : remaining = 0
  · refine ⟨[], ?_⟩
    rw [powerLoop, if_pos finished]
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false) (Evaluates.lookup locals.getExponent)
      (by simp [Mutable.truth, scalarTruth, finished])
    apply BlockReturns.return_value
    exact Evaluates.tuple (.cons (Evaluates.lookup locals.getLower)
      (.cons (Evaluates.lookup locals.getUpper) (.cons (Evaluates.lookup locals.getShift) .nil))) (by simp)
  · obtain ⟨nextEnv, garbage, executed, nextLocals⟩ := step env heap globals precision remaining a f locals
    obtain ⟨extra, rest⟩ := loop_result precision (remaining / 2)
      (nextAccumulator precision remaining a f) (nextFactor precision remaining f)
      nextEnv (heap ++ garbage) globals nextLocals
    have answer : powerLoop precision a f remaining =
        powerLoop precision (nextAccumulator precision remaining a f) (nextFactor precision remaining f)
          (remaining / 2) := by
      rw [powerLoop, if_neg finished]
      by_cases done : remaining / 2 = 0
      · simp [done, powerLoop, nextAccumulator]
      · simp only [nextAccumulator, nextFactor, if_neg done]
    refine ⟨garbage ++ extra, ?_⟩
    apply BlockReturns.while_loop (flag := true) (Evaluates.lookup locals.getExponent)
      (by simp [Mutable.truth, scalarTruth, finished])
    apply executed.returns
    change BlockReturns program nextEnv [loop, returned] (heap ++ garbage)
      (.ref (heap ++ (garbage ++ extra)).length)
      ((heap ++ (garbage ++ extra)) ++ [boundsObject (powerLoop precision a f remaining)]) globals
    simpa only [answer, List.append_assoc] using rest
termination_by remaining
decreasing_by exact Nat.div_lt_self (by omega) (by decide)

theorem result (base exponent precision : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "power_bounds" [number base, number exponent, number precision] heap
      (.ref (heap ++ extra).length) ((heap ++ extra) ++ [boundsObject (power base exponent precision)]) globals := by
  let args := ((findFunctionIn program.body "power_bounds").getD ([], [])).1
  let source := ((findFunctionIn program.body "power_bounds").getD ([], [])).2
  let entry := functionEnv args source [number base, number exponent, number precision] globals
  let initialized := setLocal (setLocal (setLocal entry "lower" (number 1)) "upper" (number 1)) "shift" (number 0)
  let first := trim (exact base) precision
  let ready := assignNames initialized factorNames (values first)
  have locals : Locals ready precision exponent (exact 1) first := by
    constructor <;> cases globals <;> rfl
  obtain ⟨extra, ran⟩ := loop_result precision exponent (exact 1) first ready
    (heap ++ [boundsObject first]) globals locals
  refine ⟨[boundsObject first] ++ extra, ?_⟩
  apply BlockReturns.function (args := args) (body := source) rfl rfl
  apply BlockReturns.assign (value := number 1) (middle := heap) Evaluates.constant
  apply BlockReturns.assign (value := number 1) (middle := heap) Evaluates.constant
  apply BlockReturns.assign (value := number 0) (middle := heap) Evaluates.constant
  apply BlockReturns.unpack (items := values first) (afterUnpack := heap ++ [boundsObject first]) rfl
  · apply Evaluates.call _ (round_power_bounds.result (exact base) precision heap globals)
    exact .cons (Evaluates.lookup (by cases globals <;> rfl))
      (.cons (Evaluates.lookup (by cases globals <;> rfl))
        (.cons Evaluates.constant (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil)))
  · simp [Mutable.sequence, boundsObject, values, first]
  · rfl
  · simpa only [power, first, ready, initialized, entry, source, args, List.append_assoc] using ran

end solvers.«99».power_bounds
