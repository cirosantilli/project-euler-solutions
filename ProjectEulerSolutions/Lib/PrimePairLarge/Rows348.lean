import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11593_small : List Nat := []
theorem row11593_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11593 row11593_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11593_block_005 : List Nat := []
theorem row11593_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11593 row11593_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11593_block_006 : List Nat := [11833]
def plan11593_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .second, (.firstFactor 29), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row11593_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11593 row11593_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11593_block_006
  · decide
  · rfl

def row11593_block_007 : List Nat := [13567, 14323]
def plan11593_block_007 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 37), .second, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 19), .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 23), .second, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11593_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11593 row11593_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11593_block_007
  · decide
  · rfl

def row11593_block_008 : List Nat := []
theorem row11593_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11593 row11593_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11593_block_009 : List Nat := []
theorem row11593_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11593 row11593_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11593_block_010 : List Nat := []
theorem row11593_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11593 row11593_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11593_block_011 : List Nat := []
theorem row11593_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11593 row11593_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11593_block_012 : List Nat := []
theorem row11593_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11593 row11593_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11593_block_013 : List Nat := []
theorem row11593_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11593 row11593_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11593_large_left_left : List Nat := row11593_small ++ row11593_block_005
theorem row11593_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11593 row11593_large_left_left :=
  Budget.RowUpperBound.branch row11593_small_covers row11593_block_005_covers

def row11593_large_left_right_right : List Nat := row11593_block_007 ++ row11593_block_008
theorem row11593_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11593 row11593_large_left_right_right :=
  Budget.RowUpperBound.branch row11593_block_007_covers row11593_block_008_covers

def row11593_large_left_right : List Nat := row11593_block_006 ++ row11593_large_left_right_right
theorem row11593_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11593 row11593_large_left_right :=
  Budget.RowUpperBound.branch row11593_block_006_covers row11593_large_left_right_right_covers

def row11593_large_left : List Nat := row11593_large_left_left ++ row11593_large_left_right
theorem row11593_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11593 row11593_large_left :=
  Budget.RowUpperBound.branch row11593_large_left_left_covers row11593_large_left_right_covers

def row11593_large_right_left : List Nat := row11593_block_009 ++ row11593_block_010
theorem row11593_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11593 row11593_large_right_left :=
  Budget.RowUpperBound.branch row11593_block_009_covers row11593_block_010_covers

def row11593_large_right_right_right : List Nat := row11593_block_012 ++ row11593_block_013
theorem row11593_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11593 row11593_large_right_right_right :=
  Budget.RowUpperBound.branch row11593_block_012_covers row11593_block_013_covers

def row11593_large_right_right : List Nat := row11593_block_011 ++ row11593_large_right_right_right
theorem row11593_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11593 row11593_large_right_right :=
  Budget.RowUpperBound.branch row11593_block_011_covers row11593_large_right_right_right_covers

def row11593_large_right : List Nat := row11593_large_right_left ++ row11593_large_right_right
theorem row11593_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11593 row11593_large_right :=
  Budget.RowUpperBound.branch row11593_large_right_left_covers row11593_large_right_right_covers

def row11593_large : List Nat := row11593_large_left ++ row11593_large_right
theorem row11593_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11593 row11593_large :=
  Budget.RowUpperBound.branch row11593_large_left_covers row11593_large_right_covers

def row11593 : List Nat := row11593_large
theorem row11593_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11593 row11593 := row11593_large_covers

def row11597_small : List Nat := []
theorem row11597_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11597 row11597_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11597_block_005 : List Nat := []
theorem row11597_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11597 row11597_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11597_block_006 : List Nat := [11897, 12269, 12413]
def plan11597_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 17), (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 17), .first, .possible]
set_option maxRecDepth 2048 in
theorem row11597_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11597 row11597_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11597_block_006
  · decide
  · rfl

def row11597_block_007 : List Nat := [12713, 13619, 14009, 14327]
def plan11597_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 37), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .second, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11597_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11597 row11597_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11597_block_007
  · decide
  · rfl

def row11597_block_008 : List Nat := []
theorem row11597_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11597 row11597_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11597_block_009 : List Nat := []
theorem row11597_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11597 row11597_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11597_block_010 : List Nat := []
theorem row11597_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11597 row11597_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11597_block_011 : List Nat := []
theorem row11597_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11597 row11597_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11597_block_012 : List Nat := []
theorem row11597_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11597 row11597_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11597_block_013 : List Nat := []
theorem row11597_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11597 row11597_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11597_large_left_left : List Nat := row11597_small ++ row11597_block_005
theorem row11597_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11597 row11597_large_left_left :=
  Budget.RowUpperBound.branch row11597_small_covers row11597_block_005_covers

def row11597_large_left_right_right : List Nat := row11597_block_007 ++ row11597_block_008
theorem row11597_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11597 row11597_large_left_right_right :=
  Budget.RowUpperBound.branch row11597_block_007_covers row11597_block_008_covers

def row11597_large_left_right : List Nat := row11597_block_006 ++ row11597_large_left_right_right
theorem row11597_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11597 row11597_large_left_right :=
  Budget.RowUpperBound.branch row11597_block_006_covers row11597_large_left_right_right_covers

def row11597_large_left : List Nat := row11597_large_left_left ++ row11597_large_left_right
theorem row11597_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11597 row11597_large_left :=
  Budget.RowUpperBound.branch row11597_large_left_left_covers row11597_large_left_right_covers

def row11597_large_right_left : List Nat := row11597_block_009 ++ row11597_block_010
theorem row11597_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11597 row11597_large_right_left :=
  Budget.RowUpperBound.branch row11597_block_009_covers row11597_block_010_covers

def row11597_large_right_right_right : List Nat := row11597_block_012 ++ row11597_block_013
theorem row11597_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11597 row11597_large_right_right_right :=
  Budget.RowUpperBound.branch row11597_block_012_covers row11597_block_013_covers

def row11597_large_right_right : List Nat := row11597_block_011 ++ row11597_large_right_right_right
theorem row11597_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11597 row11597_large_right_right :=
  Budget.RowUpperBound.branch row11597_block_011_covers row11597_large_right_right_right_covers

def row11597_large_right : List Nat := row11597_large_right_left ++ row11597_large_right_right
theorem row11597_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11597 row11597_large_right :=
  Budget.RowUpperBound.branch row11597_large_right_left_covers row11597_large_right_right_covers

def row11597_large : List Nat := row11597_large_left ++ row11597_large_right
theorem row11597_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11597 row11597_large :=
  Budget.RowUpperBound.branch row11597_large_left_covers row11597_large_right_covers

def row11597 : List Nat := row11597_large
theorem row11597_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11597 row11597 := row11597_large_covers

def row11617_small : List Nat := []
theorem row11617_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11617 row11617_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11617_block_005 : List Nat := []
theorem row11617_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11617 row11617_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11617_block_006 : List Nat := [12043, 12157]
def plan11617_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .possible, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row11617_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11617 row11617_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11617_block_006
  · decide
  · rfl

def row11617_block_007 : List Nat := [13669, 13681, 14293]
def plan11617_block_007 : List Budget.Choice := [(.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .second, (.firstFactor 19), (.secondFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .possible, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11617_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11617 row11617_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11617_block_007
  · decide
  · rfl

def row11617_block_008 : List Nat := []
theorem row11617_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11617 row11617_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11617_block_009 : List Nat := []
theorem row11617_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11617 row11617_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11617_block_010 : List Nat := []
theorem row11617_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11617 row11617_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11617_block_011 : List Nat := []
theorem row11617_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11617 row11617_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11617_block_012 : List Nat := []
theorem row11617_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11617 row11617_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11617_block_013 : List Nat := []
theorem row11617_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11617 row11617_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11617_large_left_left : List Nat := row11617_small ++ row11617_block_005
theorem row11617_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11617 row11617_large_left_left :=
  Budget.RowUpperBound.branch row11617_small_covers row11617_block_005_covers

def row11617_large_left_right_right : List Nat := row11617_block_007 ++ row11617_block_008
theorem row11617_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11617 row11617_large_left_right_right :=
  Budget.RowUpperBound.branch row11617_block_007_covers row11617_block_008_covers

def row11617_large_left_right : List Nat := row11617_block_006 ++ row11617_large_left_right_right
theorem row11617_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11617 row11617_large_left_right :=
  Budget.RowUpperBound.branch row11617_block_006_covers row11617_large_left_right_right_covers

def row11617_large_left : List Nat := row11617_large_left_left ++ row11617_large_left_right
theorem row11617_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11617 row11617_large_left :=
  Budget.RowUpperBound.branch row11617_large_left_left_covers row11617_large_left_right_covers

def row11617_large_right_left : List Nat := row11617_block_009 ++ row11617_block_010
theorem row11617_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11617 row11617_large_right_left :=
  Budget.RowUpperBound.branch row11617_block_009_covers row11617_block_010_covers

def row11617_large_right_right_right : List Nat := row11617_block_012 ++ row11617_block_013
theorem row11617_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11617 row11617_large_right_right_right :=
  Budget.RowUpperBound.branch row11617_block_012_covers row11617_block_013_covers

def row11617_large_right_right : List Nat := row11617_block_011 ++ row11617_large_right_right_right
theorem row11617_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11617 row11617_large_right_right :=
  Budget.RowUpperBound.branch row11617_block_011_covers row11617_large_right_right_right_covers

def row11617_large_right : List Nat := row11617_large_right_left ++ row11617_large_right_right
theorem row11617_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11617 row11617_large_right :=
  Budget.RowUpperBound.branch row11617_large_right_left_covers row11617_large_right_right_covers

def row11617_large : List Nat := row11617_large_left ++ row11617_large_right
theorem row11617_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11617 row11617_large :=
  Budget.RowUpperBound.branch row11617_large_left_covers row11617_large_right_covers

def row11617 : List Nat := row11617_large
theorem row11617_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11617 row11617 := row11617_large_covers

def row11621_small : List Nat := []
theorem row11621_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11621 row11621_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11621_block_005 : List Nat := []
theorem row11621_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11621 row11621_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11621_block_006 : List Nat := []
def plan11621_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 7), .first, (.secondFactor 23), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11621_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11621 row11621_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11621_block_006
  · decide
  · rfl

def row11621_block_007 : List Nat := [14159, 14321]
def plan11621_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 37), (.firstFactor 23), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .second, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 29), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11621_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11621 row11621_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11621_block_007
  · decide
  · rfl

def row11621_block_008 : List Nat := []
theorem row11621_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11621 row11621_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11621_block_009 : List Nat := []
theorem row11621_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11621 row11621_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11621_block_010 : List Nat := []
theorem row11621_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11621 row11621_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11621_block_011 : List Nat := []
theorem row11621_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11621 row11621_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11621_block_012 : List Nat := []
theorem row11621_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11621 row11621_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11621_block_013 : List Nat := []
theorem row11621_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11621 row11621_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11621_large_left_left : List Nat := row11621_small ++ row11621_block_005
theorem row11621_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11621 row11621_large_left_left :=
  Budget.RowUpperBound.branch row11621_small_covers row11621_block_005_covers

def row11621_large_left_right_right : List Nat := row11621_block_007 ++ row11621_block_008
theorem row11621_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11621 row11621_large_left_right_right :=
  Budget.RowUpperBound.branch row11621_block_007_covers row11621_block_008_covers

def row11621_large_left_right : List Nat := row11621_block_006 ++ row11621_large_left_right_right
theorem row11621_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11621 row11621_large_left_right :=
  Budget.RowUpperBound.branch row11621_block_006_covers row11621_large_left_right_right_covers

def row11621_large_left : List Nat := row11621_large_left_left ++ row11621_large_left_right
theorem row11621_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11621 row11621_large_left :=
  Budget.RowUpperBound.branch row11621_large_left_left_covers row11621_large_left_right_covers

def row11621_large_right_left : List Nat := row11621_block_009 ++ row11621_block_010
theorem row11621_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11621 row11621_large_right_left :=
  Budget.RowUpperBound.branch row11621_block_009_covers row11621_block_010_covers

def row11621_large_right_right_right : List Nat := row11621_block_012 ++ row11621_block_013
theorem row11621_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11621 row11621_large_right_right_right :=
  Budget.RowUpperBound.branch row11621_block_012_covers row11621_block_013_covers

def row11621_large_right_right : List Nat := row11621_block_011 ++ row11621_large_right_right_right
theorem row11621_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11621 row11621_large_right_right :=
  Budget.RowUpperBound.branch row11621_block_011_covers row11621_large_right_right_right_covers

def row11621_large_right : List Nat := row11621_large_right_left ++ row11621_large_right_right
theorem row11621_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11621 row11621_large_right :=
  Budget.RowUpperBound.branch row11621_large_right_left_covers row11621_large_right_right_covers

def row11621_large : List Nat := row11621_large_left ++ row11621_large_right
theorem row11621_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11621 row11621_large :=
  Budget.RowUpperBound.branch row11621_large_left_covers row11621_large_right_covers

def row11621 : List Nat := row11621_large
theorem row11621_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11621 row11621 := row11621_large_covers

end ProjectEulerSolutions.PrimePairLarge
