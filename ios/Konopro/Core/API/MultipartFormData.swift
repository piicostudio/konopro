import Foundation

struct MultipartFormData {
    let boundary: String
    private(set) var body = Data()

    init(boundary: String = "Boundary-\(UUID().uuidString)") {
        self.boundary = boundary
    }

    mutating func addField(name: String, value: String) {
        appendBoundary()
        body.appendString("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        body.appendString(value)
        body.appendString("\r\n")
    }

    mutating func addFile(
        name: String,
        filename: String,
        contentType: String,
        data: Data
    ) {
        appendBoundary()
        body.appendString(
            "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n"
        )
        body.appendString("Content-Type: \(contentType)\r\n\r\n")
        body.append(data)
        body.appendString("\r\n")
    }

    mutating func finalize() -> Data {
        body.appendString("--\(boundary)--\r\n")
        return body
    }

    var contentType: String {
        "multipart/form-data; boundary=\(boundary)"
    }

    private mutating func appendBoundary() {
        body.appendString("--\(boundary)\r\n")
    }
}

private extension Data {
    mutating func appendString(_ value: String) {
        append(Data(value.utf8))
    }
}

