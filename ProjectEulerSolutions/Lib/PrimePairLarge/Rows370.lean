import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12421_small : List Nat := []
theorem row12421_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12421 row12421_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12421_block_005 : List Nat := []
theorem row12421_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12421 row12421_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12421_block_006 : List Nat := []
theorem row12421_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12421 row12421_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12421_block_007 : List Nat := [12637, 12841, 13339]
def plan12421_block_007 : List Budget.Choice := [.outside, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .possible, (.secondFactor 17), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .second, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), (.firstFactor 29), .second, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12421_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12421 row12421_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12421_block_007
  · decide
  · rfl

def row12421_block_008 : List Nat := []
theorem row12421_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12421 row12421_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12421_block_009 : List Nat := []
theorem row12421_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12421 row12421_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12421_block_010 : List Nat := []
theorem row12421_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12421 row12421_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12421_block_011 : List Nat := []
theorem row12421_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12421 row12421_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12421_block_012 : List Nat := []
theorem row12421_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12421 row12421_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12421_block_013 : List Nat := []
theorem row12421_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12421 row12421_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12421_large_left_left : List Nat := row12421_small ++ row12421_block_005
theorem row12421_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12421 row12421_large_left_left :=
  Budget.RowUpperBound.branch row12421_small_covers row12421_block_005_covers

def row12421_large_left_right_right : List Nat := row12421_block_007 ++ row12421_block_008
theorem row12421_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12421 row12421_large_left_right_right :=
  Budget.RowUpperBound.branch row12421_block_007_covers row12421_block_008_covers

def row12421_large_left_right : List Nat := row12421_block_006 ++ row12421_large_left_right_right
theorem row12421_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12421 row12421_large_left_right :=
  Budget.RowUpperBound.branch row12421_block_006_covers row12421_large_left_right_right_covers

def row12421_large_left : List Nat := row12421_large_left_left ++ row12421_large_left_right
theorem row12421_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12421 row12421_large_left :=
  Budget.RowUpperBound.branch row12421_large_left_left_covers row12421_large_left_right_covers

def row12421_large_right_left : List Nat := row12421_block_009 ++ row12421_block_010
theorem row12421_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12421 row12421_large_right_left :=
  Budget.RowUpperBound.branch row12421_block_009_covers row12421_block_010_covers

def row12421_large_right_right_right : List Nat := row12421_block_012 ++ row12421_block_013
theorem row12421_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12421 row12421_large_right_right_right :=
  Budget.RowUpperBound.branch row12421_block_012_covers row12421_block_013_covers

def row12421_large_right_right : List Nat := row12421_block_011 ++ row12421_large_right_right_right
theorem row12421_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12421 row12421_large_right_right :=
  Budget.RowUpperBound.branch row12421_block_011_covers row12421_large_right_right_right_covers

def row12421_large_right : List Nat := row12421_large_right_left ++ row12421_large_right_right
theorem row12421_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12421 row12421_large_right :=
  Budget.RowUpperBound.branch row12421_large_right_left_covers row12421_large_right_right_covers

def row12421_large : List Nat := row12421_large_left ++ row12421_large_right
theorem row12421_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12421 row12421_large :=
  Budget.RowUpperBound.branch row12421_large_left_covers row12421_large_right_covers

def row12421 : List Nat := row12421_large
theorem row12421_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12421 row12421 := row12421_large_covers

def row12433_small : List Nat := []
theorem row12433_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12433 row12433_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12433_block_005 : List Nat := []
theorem row12433_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12433 row12433_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12433_block_006 : List Nat := []
theorem row12433_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12433 row12433_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12433_block_007 : List Nat := [12619, 12907, 13033, 13177, 13597]
def plan12433_block_007 : List Budget.Choice := [.outside, .outside, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .second, (.secondFactor 19), .first, (.secondFactor 17), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .possible, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 17), .second, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12433_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12433 row12433_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12433_block_007
  · decide
  · rfl

def row12433_block_008 : List Nat := []
theorem row12433_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12433 row12433_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12433_block_009 : List Nat := []
theorem row12433_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12433 row12433_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12433_block_010 : List Nat := []
theorem row12433_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12433 row12433_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12433_block_011 : List Nat := []
theorem row12433_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12433 row12433_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12433_block_012 : List Nat := []
theorem row12433_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12433 row12433_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12433_block_013 : List Nat := []
theorem row12433_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12433 row12433_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12433_large_left_left : List Nat := row12433_small ++ row12433_block_005
theorem row12433_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12433 row12433_large_left_left :=
  Budget.RowUpperBound.branch row12433_small_covers row12433_block_005_covers

def row12433_large_left_right_right : List Nat := row12433_block_007 ++ row12433_block_008
theorem row12433_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12433 row12433_large_left_right_right :=
  Budget.RowUpperBound.branch row12433_block_007_covers row12433_block_008_covers

def row12433_large_left_right : List Nat := row12433_block_006 ++ row12433_large_left_right_right
theorem row12433_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12433 row12433_large_left_right :=
  Budget.RowUpperBound.branch row12433_block_006_covers row12433_large_left_right_right_covers

def row12433_large_left : List Nat := row12433_large_left_left ++ row12433_large_left_right
theorem row12433_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12433 row12433_large_left :=
  Budget.RowUpperBound.branch row12433_large_left_left_covers row12433_large_left_right_covers

def row12433_large_right_left : List Nat := row12433_block_009 ++ row12433_block_010
theorem row12433_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12433 row12433_large_right_left :=
  Budget.RowUpperBound.branch row12433_block_009_covers row12433_block_010_covers

def row12433_large_right_right_right : List Nat := row12433_block_012 ++ row12433_block_013
theorem row12433_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12433 row12433_large_right_right_right :=
  Budget.RowUpperBound.branch row12433_block_012_covers row12433_block_013_covers

def row12433_large_right_right : List Nat := row12433_block_011 ++ row12433_large_right_right_right
theorem row12433_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12433 row12433_large_right_right :=
  Budget.RowUpperBound.branch row12433_block_011_covers row12433_large_right_right_right_covers

def row12433_large_right : List Nat := row12433_large_right_left ++ row12433_large_right_right
theorem row12433_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12433 row12433_large_right :=
  Budget.RowUpperBound.branch row12433_large_right_left_covers row12433_large_right_right_covers

def row12433_large : List Nat := row12433_large_left ++ row12433_large_right
theorem row12433_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12433 row12433_large :=
  Budget.RowUpperBound.branch row12433_large_left_covers row12433_large_right_covers

def row12433 : List Nat := row12433_large
theorem row12433_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12433 row12433 := row12433_large_covers

def row12437_small : List Nat := []
theorem row12437_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12437 row12437_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12437_block_005 : List Nat := []
theorem row12437_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12437 row12437_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12437_block_006 : List Nat := []
theorem row12437_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12437 row12437_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12437_block_007 : List Nat := [13127, 13241, 13487]
def plan12437_block_007 : List Budget.Choice := [.outside, .outside, .outside, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 31), (.firstFactor 7), .second, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 17), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .possible, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12437_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12437 row12437_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12437_block_007
  · decide
  · rfl

def row12437_block_008 : List Nat := []
theorem row12437_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12437 row12437_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12437_block_009 : List Nat := []
theorem row12437_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12437 row12437_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12437_block_010 : List Nat := []
theorem row12437_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12437 row12437_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12437_block_011 : List Nat := []
theorem row12437_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12437 row12437_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12437_block_012 : List Nat := []
theorem row12437_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12437 row12437_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12437_block_013 : List Nat := []
theorem row12437_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12437 row12437_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12437_large_left_left : List Nat := row12437_small ++ row12437_block_005
theorem row12437_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12437 row12437_large_left_left :=
  Budget.RowUpperBound.branch row12437_small_covers row12437_block_005_covers

def row12437_large_left_right_right : List Nat := row12437_block_007 ++ row12437_block_008
theorem row12437_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12437 row12437_large_left_right_right :=
  Budget.RowUpperBound.branch row12437_block_007_covers row12437_block_008_covers

def row12437_large_left_right : List Nat := row12437_block_006 ++ row12437_large_left_right_right
theorem row12437_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12437 row12437_large_left_right :=
  Budget.RowUpperBound.branch row12437_block_006_covers row12437_large_left_right_right_covers

def row12437_large_left : List Nat := row12437_large_left_left ++ row12437_large_left_right
theorem row12437_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12437 row12437_large_left :=
  Budget.RowUpperBound.branch row12437_large_left_left_covers row12437_large_left_right_covers

def row12437_large_right_left : List Nat := row12437_block_009 ++ row12437_block_010
theorem row12437_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12437 row12437_large_right_left :=
  Budget.RowUpperBound.branch row12437_block_009_covers row12437_block_010_covers

def row12437_large_right_right_right : List Nat := row12437_block_012 ++ row12437_block_013
theorem row12437_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12437 row12437_large_right_right_right :=
  Budget.RowUpperBound.branch row12437_block_012_covers row12437_block_013_covers

def row12437_large_right_right : List Nat := row12437_block_011 ++ row12437_large_right_right_right
theorem row12437_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12437 row12437_large_right_right :=
  Budget.RowUpperBound.branch row12437_block_011_covers row12437_large_right_right_right_covers

def row12437_large_right : List Nat := row12437_large_right_left ++ row12437_large_right_right
theorem row12437_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12437 row12437_large_right :=
  Budget.RowUpperBound.branch row12437_large_right_left_covers row12437_large_right_right_covers

def row12437_large : List Nat := row12437_large_left ++ row12437_large_right
theorem row12437_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12437 row12437_large :=
  Budget.RowUpperBound.branch row12437_large_left_covers row12437_large_right_covers

def row12437 : List Nat := row12437_large
theorem row12437_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12437 row12437 := row12437_large_covers

def row12451_small : List Nat := []
theorem row12451_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12451 row12451_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12451_block_005 : List Nat := []
theorem row12451_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12451 row12451_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12451_block_006 : List Nat := []
theorem row12451_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12451 row12451_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12451_block_007 : List Nat := []
def plan12451_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 31), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12451_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12451 row12451_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12451_block_007
  · decide
  · rfl

def row12451_block_008 : List Nat := []
theorem row12451_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12451 row12451_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12451_block_009 : List Nat := []
theorem row12451_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12451 row12451_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12451_block_010 : List Nat := []
theorem row12451_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12451 row12451_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12451_block_011 : List Nat := []
theorem row12451_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12451 row12451_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12451_block_012 : List Nat := []
theorem row12451_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12451 row12451_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12451_block_013 : List Nat := []
theorem row12451_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12451 row12451_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12451_large_left_left : List Nat := row12451_small ++ row12451_block_005
theorem row12451_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12451 row12451_large_left_left :=
  Budget.RowUpperBound.branch row12451_small_covers row12451_block_005_covers

def row12451_large_left_right_right : List Nat := row12451_block_007 ++ row12451_block_008
theorem row12451_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12451 row12451_large_left_right_right :=
  Budget.RowUpperBound.branch row12451_block_007_covers row12451_block_008_covers

def row12451_large_left_right : List Nat := row12451_block_006 ++ row12451_large_left_right_right
theorem row12451_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12451 row12451_large_left_right :=
  Budget.RowUpperBound.branch row12451_block_006_covers row12451_large_left_right_right_covers

def row12451_large_left : List Nat := row12451_large_left_left ++ row12451_large_left_right
theorem row12451_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12451 row12451_large_left :=
  Budget.RowUpperBound.branch row12451_large_left_left_covers row12451_large_left_right_covers

def row12451_large_right_left : List Nat := row12451_block_009 ++ row12451_block_010
theorem row12451_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12451 row12451_large_right_left :=
  Budget.RowUpperBound.branch row12451_block_009_covers row12451_block_010_covers

def row12451_large_right_right_right : List Nat := row12451_block_012 ++ row12451_block_013
theorem row12451_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12451 row12451_large_right_right_right :=
  Budget.RowUpperBound.branch row12451_block_012_covers row12451_block_013_covers

def row12451_large_right_right : List Nat := row12451_block_011 ++ row12451_large_right_right_right
theorem row12451_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12451 row12451_large_right_right :=
  Budget.RowUpperBound.branch row12451_block_011_covers row12451_large_right_right_right_covers

def row12451_large_right : List Nat := row12451_large_right_left ++ row12451_large_right_right
theorem row12451_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12451 row12451_large_right :=
  Budget.RowUpperBound.branch row12451_large_right_left_covers row12451_large_right_right_covers

def row12451_large : List Nat := row12451_large_left ++ row12451_large_right
theorem row12451_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12451 row12451_large :=
  Budget.RowUpperBound.branch row12451_large_left_covers row12451_large_right_covers

def row12451 : List Nat := row12451_large
theorem row12451_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12451 row12451 := row12451_large_covers

end ProjectEulerSolutions.PrimePairLarge
