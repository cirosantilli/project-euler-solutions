import ProjectEulerStatements.Lib.TotientLinearCache

set_option linter.style.header false

/-! A linear totient sieve with a live entry array and an indexed prime cache. -/
namespace ProjectEulerSolutions.TotientArraySieve
open ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false

/-- The array stores the totient and composite flag of each in-range integer. -/
def encode (size : Nat) (table : Table) : Array Cell :=
  Array.ofFn (fun i : Fin size => table i)

@[simp] theorem encode_size (size : Nat) (table : Table) : (encode size table).size = size := by
  simp [encode]

theorem encode_read (size i : Nat) (table : Table) (within : i < size) :
    (encode size table)[i]! = table i := by
  simp [encode, getElem!_pos, within]

theorem encode_set (size i : Nat) (table : Table) (replacement : Cell) :
    (encode size table).set! i replacement = encode size (Function.update table i replacement) := by
  apply Array.ext
  · simp
  · intro j leftBound rightBound
    have bounded : j < size := by simpa using rightBound
    simp only [Array.set!_eq_setIfInBounds]
    rw [Array.getElem_setIfInBounds (by simpa using bounded)]
    by_cases same : i = j
    · subst j
      simp [encode]
    · simp [encode, same, Function.update_of_ne (Ne.symm same)]

/-- The product cell is marked in the same update which stores its totient. -/
def writeCell (i p : Nat) (cells : Array Cell) : Array Cell :=
  cells.set! (i * p)
    (if i % p = 0 then ((cells[i]!).1 * p, true) else ((cells[i]!).1 * (p - 1), true))

theorem writeCell_encode (limit i p : Nat) (table : Table) (within : i ≤ limit) :
    writeCell i p (encode (limit + 1) table) = encode (limit + 1) (write i p table) := by
  simp only [writeCell, encode_read (limit + 1) i table (by omega), write]
  exact encode_set (limit + 1) (i * p) table _

/-- The cache is traversed by index; proof-side list conversions do not run in the sieve. -/
def passArray (limit i : Nat) (cache : Array Nat) (position : Nat) (cells : Array Cell) :
    Array Cell :=
  if h : position < cache.size then
    let p := cache[position]
    if limit < i * p then cells
    else if i % p = 0 then writeCell i p cells
    else passArray limit i cache (position + 1) (writeCell i p cells)
  else cells
termination_by cache.size - position
decreasing_by omega

/-- A false composite flag initializes a prime cell and appends to the live cache. -/
def prepareArray (i : Nat) (cells : Array Cell) (cache : Array Nat) : Array Cell × Array Nat :=
  if (cells[i]!).2 then (cells, cache) else (cells.set! i (i - 1, false), cache.push i)

def runArray (limit i : Nat) (cells : Array Cell) (cache : Array Nat) : Array Cell :=
  if i ≤ limit then
    let (prepared, cache) := prepareArray i cells cache
    let updated := passArray limit i cache 0 prepared
    runArray limit (i + 1) updated cache
  else cells
termination_by limit + 1 - i
decreasing_by omega

def initialCells (limit : Nat) : Array Cell :=
  (Array.replicate (limit + 1) (0, false)).set! 1 (1, false)

def computeCells (limit : Nat) : Array Cell := runArray limit 2 (initialCells limit) #[]

def computePhiLinear (limit : Nat) : Array Nat := (computeCells limit).map Prod.fst

end ProjectEulerSolutions.TotientArraySieve
