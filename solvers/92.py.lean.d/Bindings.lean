import solvers.«92.py.lean.d».Rows
import lang.py.FormalSW.UnpackingCalls

namespace solvers.«92».bindings
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev target : Expr := .tuple [.tuple [.name "total", .name "tight"], .name "ways"]
abbrev body : List Stmt := [.assign [target] (.name "$formalsw.unpack")]

/-- The dictionary iterator supplies the original key and its current count. -/
def locals (before : HeapCalls.Env) (keyAddress ways : Nat) (state : Prefix.State) : HeapCalls.Env :=
  let outer := assignNames before ["$formalsw.target.0", "$formalsw.target.1"]
    [.ref keyAddress, table.number ways]
  let inner := assignNames outer ["$formalsw.target.0.0", "$formalsw.target.0.1"]
    [table.number state.1, .scalar (.bool state.2)]
  setLocal (setLocal (setLocal inner "total" (table.number state.1)) "tight" (.scalar (.bool state.2)))
    "ways" (table.number ways)

theorem source :
    (match (executableBody ((findFunctionIn rows.program.body "solve").getD ([], [])).2)[2]? with
      | some (Stmt.forIn _ _ prefixBody _) => match prefixBody[2]? with
          | some (Stmt.forDictionaryItems item _ _ _) => item
          | _ => .name "$missing.row"
      | _ => .name "$missing.prefix") = target := rfl

/-- Unpacking neither copies either tuple nor changes any heap cell. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (pairAddress keyAddress ways : Nat) (state : Prefix.State)
    (lookup : before "$formalsw.unpack" = some (.ref pairAddress))
    (pair : heap[pairAddress]? = some (.tuple [.ref keyAddress, table.number ways]))
    (key : heap[keyAddress]? = some (table.tuple state)) :
    BlockContinues rows.program before body heap (locals before keyAddress ways state) heap globals := by
  apply BlockContinues.unpack_nested false (by rfl)
  have outer := BlockContinues.unpack (module := rows.program) (globals := globals)
    (heap := heap) (middle := heap) (finalHeap := heap)
    (items := [.ref keyAddress, table.number ways])
    (expressions := [.name "$formalsw.target.0", .name "$formalsw.target.1"])
    (names := ["$formalsw.target.0", "$formalsw.target.1"]) rfl
    (Evaluates.lookup lookup) (by simp [Mutable.sequence, pair]) (by rfl)
  have inner := BlockContinues.unpack (module := rows.program) (globals := globals)
    (heap := heap) (middle := heap) (finalHeap := heap)
    (items := [table.number state.1, .scalar (.bool state.2)]) (value := .ref keyAddress)
    (expressions := [.name "$formalsw.target.0.0", .name "$formalsw.target.0.1"])
    (names := ["$formalsw.target.0.0", "$formalsw.target.0.1"]) rfl
    (Evaluates.lookup (name := "$formalsw.target.0") (heap := heap)
      (env := assignNames before ["$formalsw.target.0", "$formalsw.target.1"]
        [.ref keyAddress, table.number ways]) (by simp [assignNames, setLocal]))
    (by simp [Mutable.sequence, key, table.tuple]) (by rfl)
  simpa [UnpackingTargets.assignment, UnpackingTargets.lower, UnpackingTargets.children,
    UnpackingTargets.names, UnpackingTargets.temporary, List.range_succ, locals] using
    outer.append (inner.append
    ((BlockContinues.assign (name := "total")
      (Evaluates.lookup (name := "$formalsw.target.0.0") rfl)).append
    ((BlockContinues.assign (name := "tight")
      (Evaluates.lookup (name := "$formalsw.target.0.1") rfl)).append
    ((BlockContinues.assign (name := "ways")
      (Evaluates.lookup (name := "$formalsw.target.1") rfl)).append BlockContinues.nil))))

def rebound : List String :=
  ["$formalsw.target.0", "$formalsw.target.1", "$formalsw.target.0.0", "$formalsw.target.0.1",
    "total", "tight", "ways"]

theorem unchanged (before : HeapCalls.Env) (keyAddress ways : Nat) (state : Prefix.State)
    (name : String) (keep : name ∉ rebound) :
    locals before keyAddress ways state name = before name := by
  simp only [rebound, List.mem_cons, not_or] at keep
  simp_all [locals, assignNames, setLocal]

theorem ready (before : HeapCalls.Env) (address keyAddress ways ceiling : Nat) (state : Prefix.State)
    (following : before "following" = some (.ref address))
    (limit : before "ceiling" = some (table.number ceiling)) :
    rows.Ready (locals before keyAddress ways state) address state ways ceiling := by
  constructor <;> simp [locals, assignNames, setLocal, following, limit]

end solvers.«92».bindings
