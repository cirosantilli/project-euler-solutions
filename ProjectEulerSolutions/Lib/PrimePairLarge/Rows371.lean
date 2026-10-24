import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12457_small : List Nat := []
theorem row12457_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12457 row12457_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12457_block_005 : List Nat := []
theorem row12457_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12457 row12457_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12457_block_006 : List Nat := []
theorem row12457_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12457 row12457_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12457_block_007 : List Nat := []
def plan12457_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, .second, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 29), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12457_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12457 row12457_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12457_block_007
  · decide
  · rfl

def row12457_block_008 : List Nat := []
theorem row12457_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12457 row12457_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12457_block_009 : List Nat := []
theorem row12457_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12457 row12457_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12457_block_010 : List Nat := []
theorem row12457_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12457 row12457_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12457_block_011 : List Nat := []
theorem row12457_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12457 row12457_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12457_block_012 : List Nat := []
theorem row12457_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12457 row12457_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12457_block_013 : List Nat := []
theorem row12457_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12457 row12457_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12457_large_left_left : List Nat := row12457_small ++ row12457_block_005
theorem row12457_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12457 row12457_large_left_left :=
  Budget.RowUpperBound.branch row12457_small_covers row12457_block_005_covers

def row12457_large_left_right_right : List Nat := row12457_block_007 ++ row12457_block_008
theorem row12457_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12457 row12457_large_left_right_right :=
  Budget.RowUpperBound.branch row12457_block_007_covers row12457_block_008_covers

def row12457_large_left_right : List Nat := row12457_block_006 ++ row12457_large_left_right_right
theorem row12457_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12457 row12457_large_left_right :=
  Budget.RowUpperBound.branch row12457_block_006_covers row12457_large_left_right_right_covers

def row12457_large_left : List Nat := row12457_large_left_left ++ row12457_large_left_right
theorem row12457_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12457 row12457_large_left :=
  Budget.RowUpperBound.branch row12457_large_left_left_covers row12457_large_left_right_covers

def row12457_large_right_left : List Nat := row12457_block_009 ++ row12457_block_010
theorem row12457_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12457 row12457_large_right_left :=
  Budget.RowUpperBound.branch row12457_block_009_covers row12457_block_010_covers

def row12457_large_right_right_right : List Nat := row12457_block_012 ++ row12457_block_013
theorem row12457_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12457 row12457_large_right_right_right :=
  Budget.RowUpperBound.branch row12457_block_012_covers row12457_block_013_covers

def row12457_large_right_right : List Nat := row12457_block_011 ++ row12457_large_right_right_right
theorem row12457_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12457 row12457_large_right_right :=
  Budget.RowUpperBound.branch row12457_block_011_covers row12457_large_right_right_right_covers

def row12457_large_right : List Nat := row12457_large_right_left ++ row12457_large_right_right
theorem row12457_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12457 row12457_large_right :=
  Budget.RowUpperBound.branch row12457_large_right_left_covers row12457_large_right_right_covers

def row12457_large : List Nat := row12457_large_left ++ row12457_large_right
theorem row12457_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12457 row12457_large :=
  Budget.RowUpperBound.branch row12457_large_left_covers row12457_large_right_covers

def row12457 : List Nat := row12457_large
theorem row12457_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12457 row12457 := row12457_large_covers

def row12473_small : List Nat := []
theorem row12473_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12473 row12473_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12473_block_005 : List Nat := []
theorem row12473_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12473 row12473_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12473_block_006 : List Nat := []
theorem row12473_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12473 row12473_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12473_block_007 : List Nat := [12713, 13217]
def plan12473_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 37), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 29), (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12473_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12473 row12473_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12473_block_007
  · decide
  · rfl

def row12473_block_008 : List Nat := []
theorem row12473_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12473 row12473_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12473_block_009 : List Nat := []
theorem row12473_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12473 row12473_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12473_block_010 : List Nat := []
theorem row12473_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12473 row12473_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12473_block_011 : List Nat := []
theorem row12473_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12473 row12473_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12473_block_012 : List Nat := []
theorem row12473_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12473 row12473_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12473_block_013 : List Nat := []
theorem row12473_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12473 row12473_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12473_large_left_left : List Nat := row12473_small ++ row12473_block_005
theorem row12473_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12473 row12473_large_left_left :=
  Budget.RowUpperBound.branch row12473_small_covers row12473_block_005_covers

def row12473_large_left_right_right : List Nat := row12473_block_007 ++ row12473_block_008
theorem row12473_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12473 row12473_large_left_right_right :=
  Budget.RowUpperBound.branch row12473_block_007_covers row12473_block_008_covers

def row12473_large_left_right : List Nat := row12473_block_006 ++ row12473_large_left_right_right
theorem row12473_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12473 row12473_large_left_right :=
  Budget.RowUpperBound.branch row12473_block_006_covers row12473_large_left_right_right_covers

def row12473_large_left : List Nat := row12473_large_left_left ++ row12473_large_left_right
theorem row12473_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12473 row12473_large_left :=
  Budget.RowUpperBound.branch row12473_large_left_left_covers row12473_large_left_right_covers

def row12473_large_right_left : List Nat := row12473_block_009 ++ row12473_block_010
theorem row12473_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12473 row12473_large_right_left :=
  Budget.RowUpperBound.branch row12473_block_009_covers row12473_block_010_covers

def row12473_large_right_right_right : List Nat := row12473_block_012 ++ row12473_block_013
theorem row12473_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12473 row12473_large_right_right_right :=
  Budget.RowUpperBound.branch row12473_block_012_covers row12473_block_013_covers

def row12473_large_right_right : List Nat := row12473_block_011 ++ row12473_large_right_right_right
theorem row12473_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12473 row12473_large_right_right :=
  Budget.RowUpperBound.branch row12473_block_011_covers row12473_large_right_right_right_covers

def row12473_large_right : List Nat := row12473_large_right_left ++ row12473_large_right_right
theorem row12473_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12473 row12473_large_right :=
  Budget.RowUpperBound.branch row12473_large_right_left_covers row12473_large_right_right_covers

def row12473_large : List Nat := row12473_large_left ++ row12473_large_right
theorem row12473_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12473 row12473_large :=
  Budget.RowUpperBound.branch row12473_large_left_covers row12473_large_right_covers

def row12473 : List Nat := row12473_large
theorem row12473_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12473 row12473 := row12473_large_covers

def row12479_small : List Nat := []
theorem row12479_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12479 row12479_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12479_block_005 : List Nat := []
theorem row12479_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12479 row12479_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12479_block_006 : List Nat := []
theorem row12479_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12479 row12479_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12479_block_007 : List Nat := [12569, 12791, 13259]
def plan12479_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 37), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 29), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12479_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12479 row12479_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12479_block_007
  · decide
  · rfl

def row12479_block_008 : List Nat := []
theorem row12479_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12479 row12479_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12479_block_009 : List Nat := []
theorem row12479_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12479 row12479_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12479_block_010 : List Nat := []
theorem row12479_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12479 row12479_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12479_block_011 : List Nat := []
theorem row12479_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12479 row12479_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12479_block_012 : List Nat := []
theorem row12479_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12479 row12479_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12479_block_013 : List Nat := []
theorem row12479_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12479 row12479_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12479_large_left_left : List Nat := row12479_small ++ row12479_block_005
theorem row12479_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12479 row12479_large_left_left :=
  Budget.RowUpperBound.branch row12479_small_covers row12479_block_005_covers

def row12479_large_left_right_right : List Nat := row12479_block_007 ++ row12479_block_008
theorem row12479_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12479 row12479_large_left_right_right :=
  Budget.RowUpperBound.branch row12479_block_007_covers row12479_block_008_covers

def row12479_large_left_right : List Nat := row12479_block_006 ++ row12479_large_left_right_right
theorem row12479_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12479 row12479_large_left_right :=
  Budget.RowUpperBound.branch row12479_block_006_covers row12479_large_left_right_right_covers

def row12479_large_left : List Nat := row12479_large_left_left ++ row12479_large_left_right
theorem row12479_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12479 row12479_large_left :=
  Budget.RowUpperBound.branch row12479_large_left_left_covers row12479_large_left_right_covers

def row12479_large_right_left : List Nat := row12479_block_009 ++ row12479_block_010
theorem row12479_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12479 row12479_large_right_left :=
  Budget.RowUpperBound.branch row12479_block_009_covers row12479_block_010_covers

def row12479_large_right_right_right : List Nat := row12479_block_012 ++ row12479_block_013
theorem row12479_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12479 row12479_large_right_right_right :=
  Budget.RowUpperBound.branch row12479_block_012_covers row12479_block_013_covers

def row12479_large_right_right : List Nat := row12479_block_011 ++ row12479_large_right_right_right
theorem row12479_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12479 row12479_large_right_right :=
  Budget.RowUpperBound.branch row12479_block_011_covers row12479_large_right_right_right_covers

def row12479_large_right : List Nat := row12479_large_right_left ++ row12479_large_right_right
theorem row12479_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12479 row12479_large_right :=
  Budget.RowUpperBound.branch row12479_large_right_left_covers row12479_large_right_right_covers

def row12479_large : List Nat := row12479_large_left ++ row12479_large_right
theorem row12479_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12479 row12479_large :=
  Budget.RowUpperBound.branch row12479_large_left_covers row12479_large_right_covers

def row12479 : List Nat := row12479_large
theorem row12479_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12479 row12479 := row12479_large_covers

def row12487_small : List Nat := []
theorem row12487_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12487 row12487_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12487_block_005 : List Nat := []
theorem row12487_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12487 row12487_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12487_block_006 : List Nat := []
theorem row12487_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12487 row12487_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12487_block_007 : List Nat := [12601, 12721, 12919, 13441]
def plan12487_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 37), (.firstFactor 13), .first, .first, .second, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 23), .second, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 37), (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12487_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12487 row12487_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12487_block_007
  · decide
  · rfl

def row12487_block_008 : List Nat := []
theorem row12487_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12487 row12487_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12487_block_009 : List Nat := []
theorem row12487_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12487 row12487_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12487_block_010 : List Nat := []
theorem row12487_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12487 row12487_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12487_block_011 : List Nat := []
theorem row12487_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12487 row12487_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12487_block_012 : List Nat := []
theorem row12487_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12487 row12487_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12487_block_013 : List Nat := []
theorem row12487_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12487 row12487_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12487_large_left_left : List Nat := row12487_small ++ row12487_block_005
theorem row12487_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12487 row12487_large_left_left :=
  Budget.RowUpperBound.branch row12487_small_covers row12487_block_005_covers

def row12487_large_left_right_right : List Nat := row12487_block_007 ++ row12487_block_008
theorem row12487_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12487 row12487_large_left_right_right :=
  Budget.RowUpperBound.branch row12487_block_007_covers row12487_block_008_covers

def row12487_large_left_right : List Nat := row12487_block_006 ++ row12487_large_left_right_right
theorem row12487_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12487 row12487_large_left_right :=
  Budget.RowUpperBound.branch row12487_block_006_covers row12487_large_left_right_right_covers

def row12487_large_left : List Nat := row12487_large_left_left ++ row12487_large_left_right
theorem row12487_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12487 row12487_large_left :=
  Budget.RowUpperBound.branch row12487_large_left_left_covers row12487_large_left_right_covers

def row12487_large_right_left : List Nat := row12487_block_009 ++ row12487_block_010
theorem row12487_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12487 row12487_large_right_left :=
  Budget.RowUpperBound.branch row12487_block_009_covers row12487_block_010_covers

def row12487_large_right_right_right : List Nat := row12487_block_012 ++ row12487_block_013
theorem row12487_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12487 row12487_large_right_right_right :=
  Budget.RowUpperBound.branch row12487_block_012_covers row12487_block_013_covers

def row12487_large_right_right : List Nat := row12487_block_011 ++ row12487_large_right_right_right
theorem row12487_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12487 row12487_large_right_right :=
  Budget.RowUpperBound.branch row12487_block_011_covers row12487_large_right_right_right_covers

def row12487_large_right : List Nat := row12487_large_right_left ++ row12487_large_right_right
theorem row12487_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12487 row12487_large_right :=
  Budget.RowUpperBound.branch row12487_large_right_left_covers row12487_large_right_right_covers

def row12487_large : List Nat := row12487_large_left ++ row12487_large_right
theorem row12487_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12487 row12487_large :=
  Budget.RowUpperBound.branch row12487_large_left_covers row12487_large_right_covers

def row12487 : List Nat := row12487_large
theorem row12487_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12487 row12487 := row12487_large_covers

end ProjectEulerSolutions.PrimePairLarge
