// Reads the text on screen in sampled frames of a video with macOS's built-in text recognition
// (Vision). Local only: nothing is sent anywhere, and nothing read from the video is printed.
//
//   ocr_video_text VIDEO OUT.json START_SECONDS END_SECONDS STEP_SECONDS
//
// OUT.json: {"frames": [{"t_ms": 12000, "items": [{"text": "...", "x": .., "y": .., "w": .., "h": ..,
// "conf": ..}]}]} with boxes in ten-thousandths of the frame, origin top-left, and confidence in
// thousandths. Whole numbers only, so the file is the same bytes on every run.
//
// scripts/read_lineups.py compiles and runs this; you do not need to run it yourself.
import Foundation
import AVFoundation
import Vision

func fail(_ message: String) -> Never {
    fputs(message + "\n", stderr)
    exit(1)
}

let args = CommandLine.arguments
guard args.count == 6, let start = Double(args[3]), let end = Double(args[4]),
      let step = Double(args[5]), step > 0, end >= start else {
    fail("usage: ocr_video_text VIDEO OUT.json START_SECONDS END_SECONDS STEP_SECONDS")
}
let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
let duration = CMTimeGetSeconds(asset.duration)
guard duration.isFinite, duration > 0 else { fail("cannot read the video: \(args[1])") }

let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.maximumSize = CGSize(width: 1920, height: 1080)
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero

func units(_ value: Double, _ scale: Double) -> Int { Int((value * scale).rounded()) }

var frames: [[String: Any]] = []
var decoded = 0
var k = 0
let last = min(end, duration - 0.1)
while start + Double(k) * step <= last {
    let t = start + Double(k) * step
    autoreleasepool {
        var actual = CMTime.zero
        guard let image = try? generator.copyCGImage(
            at: CMTime(seconds: t, preferredTimescale: 600), actualTime: &actual) else { return }
        decoded += 1
        let request = VNRecognizeTextRequest()
        request.revision = VNRecognizeTextRequestRevision3      // pinned: same recogniser every run
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false                   // names must not be "corrected"
        request.recognitionLanguages = ["es-ES", "en-US"]
        try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        var items: [[String: Any]] = []
        for observation in request.results ?? [] {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let box = observation.boundingBox                    // origin bottom-left, 0-1
            items.append(["text": candidate.string,
                          "x": units(box.minX, 10000), "y": units(1 - box.maxY, 10000),
                          "w": units(box.width, 10000), "h": units(box.height, 10000),
                          "conf": units(Double(candidate.confidence), 1000)])
        }
        frames.append(["t_ms": units(t, 1000), "items": items])
    }
    k += 1
    if k % 10 == 0 { fputs("\r  frames looked at: \(k)", stderr) }
}
fputs("\r  frames looked at: \(k)\n", stderr)
guard decoded > 0 else {
    fail("no video frame could be decoded. Run from Terminal, where macOS media services are available.")
}
let result: [String: Any] = ["method": "macOS_Vision_text_revision3_accurate_no_language_correction",
                             "start_ms": units(start, 1000), "end_ms": units(end, 1000),
                             "step_ms": units(step, 1000), "frames": frames]
do {
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    try data.write(to: URL(fileURLWithPath: args[2]), options: .atomic)
} catch {
    fail("could not write \(args[2]): \(error.localizedDescription)")
}
